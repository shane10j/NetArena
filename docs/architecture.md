# NetArena MALT Purple Agent Scaffold

This scaffold keeps the public submission surface simple: an A2A endpoint exposed by the
coordinator. The Amber root manifest can run three containers from the same image:

- `coordinator` receives benchmark traffic and exports the public `a2a` capability.
- `planner` produces concise NetworkX graph-operation plans.
- `verifier` reviews graph code for executable format, return shape, and safety.

Each role is selected with `src/server.py --role <role>`. When Amber wires the planner and verifier
slots, the coordinator delegates over A2A using `PLANNER_AGENT_URL` and `VERIFIER_AGENT_URL`.
Without those slots, the same container still runs as a standalone coordinator.

LLM access is intentionally isolated in `src/llm.py`. Configure it with `MODEL_NAME` plus the
`LITELLM_*` variables, or bind an Amber `llm` slot so the component receives `LLM_API_URL`.
If no model is configured, the agent returns a deterministic `process_graph(graph_data)` fallback
so A2A conformance checks and Docker startup still work.
