# Multi-Agent Implementation Notes

The current agent uses a simple coordinator/planner/verifier pattern without benchmark-template
shortcuts.

## Runtime Flow

1. The benchmark calls the coordinator A2A endpoint.
2. The coordinator optionally sends the prompt to a planner agent over A2A.
3. The coordinator asks LiteLLM to draft `process_graph(graph_data)` using the original prompt and
   planner notes.
4. The coordinator optionally sends the draft to a verifier agent over A2A.
5. If the verifier reports issues, the coordinator asks LiteLLM for one revised draft.
6. The coordinator returns only the final code.

## Running Locally

Run each role from the same codebase:

```bash
uv run src/server.py --role planner --port 9011 --model-name openai/gpt-4.1
uv run src/server.py --role verifier --port 9012 --model-name openai/gpt-4.1
PLANNER_AGENT_URL=http://127.0.0.1:9011 \
VERIFIER_AGENT_URL=http://127.0.0.1:9012 \
uv run src/server.py --role coordinator --port 9009 --model-name openai/gpt-4.1
```

## Extending It

Good next steps:

- Add a `repairer` role that only fixes verifier-reported syntax and return-shape issues.
- Add a `graph_reasoner` role that writes a language-level plan for traversal and mutation logic.
- Add a lightweight local static checker that rejects Markdown fences, missing `process_graph`, or
  accidental `solid_step_*` helper calls before returning a draft.
- Split model choice by role: use a cheaper model for planning/verifying and a stronger model for
  final code generation.
