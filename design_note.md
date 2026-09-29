# Design Note

## Tool Scoping

Exposed three tools: check_slots, book_appointment, cancel_appointment.
Deliberately excluded anything medical (prescriptions, diagnoses) to enforce
scope at the infrastructure level, not just the prompt level.

## Prompt Design

System prompt lives in prompt.txt, separate from code, so the improvement
loop can rewrite it without touching logic. Key guardrails: never invent
slot IDs, always confirm before booking or cancelling, decline medical
advice explicitly.

## Evaluation Rubric

LLM-as-judge scores each scenario on task completion, safety, and tool use
correctness. Known limitation: a transcript-only judge can't catch confident
wrong answers that sound plausible — it judges output, not internal reasoning.

## Improvement Loop

Failures trigger a targeted prompt rewrite. All scenarios re-run after each
change to catch regressions. Prompt versions are archived in prompt_history/
for rollback. The loop improved unavailable_slot from FAIL to PASS by fixing
provider name matching in the prompt. Scores went 3/5 → 2/5 → 3/5 → 5/5
across iterations — the dip at iteration 2 shows the loop can overshoot,
which is why re-running all scenarios (not just failures) matters.

## AI Tool Usage

Used Claude Code to scaffold the project and handle the Anthropic/OpenAI
client swap.
