
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anthropic

from tools import TOOL_IMPLEMENTATIONS, TOOL_SCHEMAS

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 2048
MAX_TOOL_ITERATIONS = 8  # guard against runaway tool-call loops

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


def load_system_prompt(path: Path = PROMPT_PATH) -> str:
    return path.read_text().strip()


class Agent:
    """A stateful multi-turn conversation wrapping the Anthropic Messages API."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        system_prompt: str | None = None,
        model: str = MODEL,
        verbose: bool = False,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.system_prompt = system_prompt if system_prompt is not None else load_system_prompt()
        self.model = model
        self.verbose = verbose
        self.messages: list[dict[str, Any]] = []
        # Structured log of every tool call made this conversation, useful
        # for eval.py to score behavior (not just final text).
        self.tool_call_log: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.messages = []
        self.tool_call_log = []

    def _log(self, *parts: Any) -> None:
        if self.verbose:
            print(*parts)

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> Any:
        impl = TOOL_IMPLEMENTATIONS.get(name)
        if impl is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            result = impl(**tool_input)
        except TypeError as exc:
            result = {"error": f"Invalid arguments for '{name}': {exc}"}
        self.tool_call_log.append({"tool": name, "input": tool_input, "result": result})
        self._log(f"  [tool] {name}({tool_input}) -> {result}")
        return result

    def send(self, user_message: str) -> str:
        """Send one user turn, run the tool loop to completion, return assistant text."""
        self.messages.append({"role": "user", "content": user_message})
        self._log(f"[user] {user_message}")

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                tools=TOOL_SCHEMAS,
                messages=self.messages,
            )

            # Always append the full assistant content (including tool_use
            # blocks) before processing, so history stays consistent.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                # Final answer: concatenate any text blocks.
                text = "".join(
                    block.text for block in response.content if block.type == "text"
                ).strip()
                self._log(f"[assistant] {text}")
                return text

            # Execute every tool_use block in this turn, collect results,
            # and send them all back in a single user message (required
            # when the model requests multiple tool calls at once).
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = self._execute_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )
            self.messages.append({"role": "user", "content": tool_results})

        # Safety valve: too many tool iterations without a final answer.
        return (
            "I'm having trouble completing that request right now — "
            "could you try rephrasing, or contact the clinic directly?"
        )

    def transcript(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable transcript (for eval.py / logging)."""
        out = []
        for msg in self.messages:
            content = msg["content"]
            if isinstance(content, str):
                out.append({"role": msg["role"], "content": content})
                continue
            rendered = []
            for block in content:
                if isinstance(block, dict):
                    rendered.append(block)
                elif block.type == "text":
                    rendered.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    rendered.append(
                        {"type": "tool_use", "name": block.name, "input": block.input}
                    )
                elif block.type == "tool_result":
                    rendered.append(
                        {"type": "tool_result", "content": block.content}
                    )
            out.append({"role": msg["role"], "content": rendered})
        return out


def run_repl() -> None:
    """Simple interactive CLI for manual testing."""
    agent = Agent(verbose=False)
    print("Scheduler agent ready. Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue
        reply = agent.send(user_input)
        print(f"Scheduler: {reply}\n")


if __name__ == "__main__":
    run_repl()
