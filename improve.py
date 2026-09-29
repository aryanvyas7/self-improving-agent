from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from agent import PROMPT_PATH, load_system_prompt
from eval import JUDGE_MODEL, print_summary, run_eval
from scenarios import SCENARIOS

HISTORY_DIR = Path(__file__).parent / "prompt_history"
LOG_PATH = Path(__file__).parent / "improve_log.json"

REVISER_MODEL = "claude-opus-4-8"

REVISER_SYSTEM_PROMPT = """You are an expert prompt engineer improving the \
system prompt for a patient appointment scheduling AI agent. You will be \
given the agent's CURRENT system prompt, a list of tools it has access to, \
and one or more FAILING test transcripts with an evaluator's explanation of \
what went wrong.

Rewrite the system prompt to fix the specific failures described, while:
  - Preserving every rule that is not implicated in a failure (don't regress \
    scenarios that already pass).
  - Keeping instructions concrete and behavioral (what to check, what order \
    to call tools in, what to confirm before acting) rather than vague \
    ("be careful").
  - Keeping the same overall style/length class as the original — this is a \
    system prompt for a production agent, not an essay.

Output ONLY the complete revised system prompt text. No preamble, no \
markdown fences, no commentary — just the prompt itself, ready to write \
directly to a file."""


def _failing_results(results: list[dict[str, Any]], score_threshold: int) -> list[dict[str, Any]]:
    failing = []
    for r in results:
        v = r["verdict"]
        if not v["passed"] or v["score"] < score_threshold:
            failing.append(r)
    return failing


def _format_failure_report(failing: list[dict[str, Any]]) -> str:
    from eval import _format_transcript_for_judge  # reuse formatting

    parts = []
    for r in failing:
        v = r["verdict"]
        parts.append(
            f"### Scenario: {r['title']} ({r['scenario_id']})\n"
            f"Success criteria: {r['success_criteria']}\n"
            f"Judge verdict: {'PASS' if v['passed'] else 'FAIL'}, score={v['score']}/5\n"
            f"Judge reasoning: {v['reasoning']}\n"
            f"Issues: {v.get('issues') or 'none listed'}\n\n"
            f"Transcript:\n{_format_transcript_for_judge(r['transcript'])}\n"
        )
    return "\n---\n".join(parts)


def propose_new_prompt(
    client: anthropic.Anthropic,
    current_prompt: str,
    failing: list[dict[str, Any]],
    model: str = REVISER_MODEL,
) -> str:
    from tools import TOOL_SCHEMAS

    tool_summary = "\n".join(f"- {t['name']}: {t['description']}" for t in TOOL_SCHEMAS)
    failure_report = _format_failure_report(failing)

    user_content = f"""CURRENT SYSTEM PROMPT:
{current_prompt}

AVAILABLE TOOLS:
{tool_summary}

FAILING SCENARIOS ({len(failing)}):
{failure_report}

Rewrite the system prompt to fix these failures."""

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=REVISER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text


def _save_prompt_version(prompt_text: str, tag: str) -> Path:
    HISTORY_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = HISTORY_DIR / f"{timestamp}_{tag}.txt"
    out_path.write_text(prompt_text)
    return out_path


def run_improvement_loop(
    max_iterations: int = 3,
    score_threshold: int = 3,
    dry_run: bool = False,
    agent_model: str | None = None,
    judge_model: str = JUDGE_MODEL,
) -> None:
    client = anthropic.Anthropic()
    log: list[dict[str, Any]] = []

    # Save the starting prompt so we always have a "before" snapshot.
    _save_prompt_version(load_system_prompt(), "baseline")

    for iteration in range(1, max_iterations + 1):
        print(f"\n=== Iteration {iteration}/{max_iterations}: running eval ===", file=sys.stderr)
        results = run_eval(scenarios=SCENARIOS, agent_model=agent_model, judge_model=judge_model)
        print_summary(results)

        failing = _failing_results(results, score_threshold)
        iteration_log = {
            "iteration": iteration,
            "passed": len(results) - len(failing),
            "total": len(results),
            "failing_scenario_ids": [r["scenario_id"] for r in failing],
        }

        if not failing:
            print("All scenarios pass (or meet the score threshold) — stopping.", file=sys.stderr)
            iteration_log["action"] = "stopped_all_passing"
            log.append(iteration_log)
            break

        print(
            f"{len(failing)} scenario(s) below threshold: "
            f"{[r['scenario_id'] for r in failing]}",
            file=sys.stderr,
        )

        current_prompt = load_system_prompt()
        new_prompt = propose_new_prompt(client, current_prompt, failing)

        if dry_run:
            print("\n--- DRY RUN: proposed new system prompt ---\n")
            print(new_prompt)
            print("\n--- end proposed prompt (not written; --dry-run) ---\n")
            iteration_log["action"] = "dry_run_proposed"
            log.append(iteration_log)
            break

        saved_path = _save_prompt_version(new_prompt, f"iter{iteration}")
        PROMPT_PATH.write_text(new_prompt)
        print(f"Wrote revised prompt to {PROMPT_PATH} (backup: {saved_path})", file=sys.stderr)
        iteration_log["action"] = "prompt_updated"
        iteration_log["prompt_backup"] = str(saved_path)
        log.append(iteration_log)
    else:
        print(f"\nReached max_iterations ({max_iterations}) without full pass.", file=sys.stderr)

    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"\nImprovement log written to {LOG_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect eval failures and iteratively improve the system prompt.")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--score-threshold", type=int, default=3, help="Scenarios scoring below this (out of 5) are treated as failing.")
    parser.add_argument("--dry-run", action="store_true", help="Propose a new prompt but don't write it or loop further.")
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    args = parser.parse_args()

    run_improvement_loop(
        max_iterations=args.max_iterations,
        score_threshold=args.score_threshold,
        dry_run=args.dry_run,
        agent_model=args.agent_model,
        judge_model=args.judge_model,
    )


if __name__ == "__main__":
    main()
