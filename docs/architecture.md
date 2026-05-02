# NetArena MALT Purple Agent Scaffold

This scaffold keeps the public submission surface simple: an A2A endpoint exposed by the
coordinator. The Amber root manifest can run four containers from the same image:

- `coordinator` receives benchmark traffic and exports the public `a2a` capability.
- `planner` produces concise NetworkX graph-operation plans.
- `proposer` drafts executable pure NetworkX code.
- `reviewer` constrains drafts to executable format, return shape, graph-copy safety, and
  helper-like behavior without private helper calls.

Each role is selected with `src/server.py --role <role>`. When Amber wires the helper slots, the
coordinator delegates over A2A using `PLANNER_AGENT_URL`, `PROPOSER_AGENT_URL`, and
`REVIEWER_AGENT_URL`.
Without those slots, the same container still runs as a standalone coordinator.

LLM access is intentionally isolated in `src/llm.py`. Configure it with `MODEL_NAME` plus the
`LITELLM_*` variables, or bind an Amber `llm` slot so the component receives `LLM_API_URL`.
If no model is configured, the agent returns a deterministic `process_graph(graph_data)` fallback
so A2A conformance checks and Docker startup still work.
