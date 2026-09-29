# Patient Appointment Scheduling Agent

A chatbot for a fake medical clinic. Checks appointment availability, books
slots, and cancels bookings using Claude's tool use. All clinic data (`tools.py`)
is in-memory Python dicts — no database, nothing persists between runs.

Includes an eval harness (5 scripted scenarios, graded by Claude as judge)
and an improvement loop that rewrites the system prompt when scenarios fail.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

Optional: `ANTHROPIC_MODEL` to override the default (`claude-sonnet-4-6`).

## Run

Chat with the agent:

```bash
python agent.py
```

Run the eval + self-improvement loop:

```bash
python improve.py
```

Just the eval, no prompt changes:

```bash
python eval.py
```

## How the improvement loop works

1. Run the 5 scenarios. Claude grades each transcript (pass/fail, score
   1–5, reasoning) against a written rubric.
2. Failing or low-scoring transcripts get bundled with the judge's
   reasoning and sent back to Claude to rewrite the prompt.
3. New prompt saved to `prompt_history/`, eval reruns. Repeats up to 3
   times (`--max-iterations`) or stops once everything passes.

More notes in `design_note.md`.
