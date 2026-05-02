# Multi-Agent Implementation Notes

The current agent uses a coordinator/planner/proposer/reviewer pattern without benchmark-template
shortcuts or private helper calls.

## Runtime Flow

1. The benchmark calls the coordinator A2A endpoint.
2. The coordinator optionally sends the prompt to a planner agent over A2A.
3. The coordinator asks a proposer agent to draft `process_graph(graph_data)` using pure NetworkX.
   If no proposer URL is configured, it uses the local LiteLLM adapter with the proposer prompt.
4. The coordinator asks a reviewer agent for feedback that constrains the code to helper-like
   behavior without calling helpers. If no reviewer URL is configured, it uses the local reviewer
   prompt when a model is available.
5. A local static review always rejects Markdown fences, missing `process_graph(graph_data)`, and
   private helper calls such as `solid_step_*`.
6. If reviewer feedback reports issues, the proposer gets one revision pass.
7. The coordinator returns only the final code, or a safe pure-NetworkX fallback if the draft still
   violates hard rules.

## Running Locally

Run each role from the same codebase:

```bash
uv run src/server.py --role planner --port 9011 --model-name openai/gpt-4.1
uv run src/server.py --role proposer --port 9012 --model-name openai/gpt-4.1
uv run src/server.py --role reviewer --port 9013 --model-name openai/gpt-4.1
PLANNER_AGENT_URL=http://127.0.0.1:9011 \
PROPOSER_AGENT_URL=http://127.0.0.1:9012 \
REVIEWER_AGENT_URL=http://127.0.0.1:9013 \
uv run 