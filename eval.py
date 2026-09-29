from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anthropic

from agent import Agent
import tools
from scenarios import SCENARIOS, Scenario

JUDGE_MODEL = "claude-opus-4-8"

RESULTS_PATH = Path(__file__).parent / "eval_results.json"

JUDGE_SYSTEM_PROMPT = """You are a strict, impartial QA evaluator for a patient \
appointment scheduling AI agent. You will be given:
  - A description of the scenario being tested and its success criteria.
  - The full conversation transcript between the (simulated) patient and the agent.
  - A log of every tool call the agent made, with inputs and results.

Judge ONLY against the stated success criteria. Be skeptical: an agent that \
sounds confident but fabricates information (e.g. claims a booking succeeded \
when the tool result shows an error, or invents a slot_id/booking_id not \
present in any tool result) must fail regardless of how polished the reply \
reads. Also fail the agent if it took an action (booking/cancelling) without \
appropriate confirmation, or if it failed to use a tool when the criteria \
required one.

Respond only with the requested structured JSON verdict."""

JUDGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {
            "type": "boolean",
            "description": "Whether the transcript satisfies the success criteria.",
        },
        "score": {
            "type": "integer",
            "description": "Quality score from 1 (fails badly) to 5 (fully meets criteria).",
        },
        "reasoning": {
            "type": "string",
            "description": "1-3 sentence explanation grounded in specific transcript/tool-log evidence.",
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific problems found, if any (empty list if none).",
        },
    },
    "required": ["passed", "score", "reasoning", "issues"],
    "additionalProperties": False,
}


def _format_transcript_for_judge(transcript: list[dict[str, Any]]) -> str:
    lines = []
    for msg in transcript:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            lines.append(f"[{role}] {content}")
            continue
        for block in content:
            btype = block.get("type")
            if btype == "text":
                lines.append(f"[{role}] {block['text']}")
            elif btype == "tool_use":
                lines.append(f"[{role} -> tool_call] {block['name']}({block['input']})")
            elif btype == "tool_result":
                lines.append(f"[tool_result] {block['content']}")
    return "\n".join(lines)


def run_scenario(scenario: Scenario, model: str | None = None, verbose: bool = False) -> dict[str, Any]:
    """Run one scenario's scripted turns against a fresh agent."""
    tools.reset_state(seed=scenario.seed)
    kwargs = {"verbose": verbose}
    if model:
        kwargs["model"] = model
    agent = Agent(**kwargs)

    replies = []
    for turn in scenario.user_turns:
        reply = agent.send(turn)
        replies.append(reply)

    return {
        "scenario_id": scenario.id,
        "title": scenario.title,
        "success_criteria": scenario.success_criteria,
        "user_turns": scenario.user_turns,
        "replies": replies,
        "transcript": agent.transcript(),
        "tool_calls": agent.tool_call_log,
    }


def judge_run(client: anthropic.Anthropic, scenario: Scenario, run: dict[str, Any], judge_model: str = JUDGE_MODEL) -> dict[str, Any]:
    """Send a completed run to Claude-as-judge and return the structured verdict."""
    transcript_text = _format_transcript_for_judge(run["transcript"])
    tool_log_text = json.dumps(run["tool_calls"], indent=2, default=str)

    user_content = f"""SCENARIO: {scenario.title}
DESCRIPTION: {scenario.description}
SUCCESS CRITERIA: {scenario.success_criteria}

TRANSCRIPT:
{transcript_text}

TOOL CALL LOG:
{tool_log_text}

Evaluate whether this transcript meets the success criteria."""

    response = client.messages.create(
        model=judge_model,
        max_tokens=1024,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        output_config={"format": {"type": "json_schema", "schema": JUDGE_OUTPUT_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def run_eval(
    scenarios: list[Scenario] = SCENARIOS,
    agent_model: str | None = None,
    judge_model: str = JUDGE_MODEL,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    client = anthropic.Anthropic()
    results = []

    for scenario in scenarios:
        print(f"Running scenario: {scenario.id} ...", file=sys.stderr)
        run = run_scenario(scenario, model=agent_model, verbose=verbose)
        verdict = judge_run(client, scenario, run, judge_model=judge_model)
        results.append({**run, "verdict": verdict})

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("EVAL SUMMARY")
    print("=" * 72)
    passed = 0
    for r in results:
        v = r["verdict"]
        status = "PASS" if v["passed"] else "FAIL"
        if v["passed"]:
            passed += 1
        print(f"[{status}] {r['scenario_id']:<24} score={v['score']}/5  {v['reasoning']}")
        for issue in v.get("issues", []):
            print(f"         - issue: {issue}")
    print("-" * 72)
    print(f"{passed}/{len(results)} scenarios passed")
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the scheduling agent against scripted scenarios.")
    parser.add_argument("--scenario", help="Run only the scenario with this id.")
    parser.add_argument("--agent-model", default=None, help="Override the agent's model.")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help="Model used as the judge.")
    parser.add_argument("--verbose", action="store_true", help="Print tool calls as they happen.")
    parser.add_argument("--out", default=str(RESULTS_PATH), help="Where to write the JSON results.")
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s.id == args.scenario]
        if not scenarios:
            print(f"No such scenario: {args.scenario}", file=sys.stderr)
            sys.exit(1)

    results = run_eval(
        scenarios=scenarios,
        agent_model=args.agent_model,
        judge_model=args.judge_model,
        verbose=args.verbose,
    )
    print_summary(results)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()
