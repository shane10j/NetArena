# NetArena MALT Purple Agent Scaffold

This scaffold keeps the public submission surface simple: an A2A endpoint exposed by the
coordinator. The Amber root manifest can run three LLM-stage role containers from the same image:

- `coordinator` receives benchmark traffic and exports the public `a2a` capability.
- `planner` produces a semantic JSON operation DSL.
- `graph_programmer` writes fallback NetworkX code only for unsupported DSL operations.
- deterministic template generation handles supported operations.
- deterministic semantic checks validate syntax, return schema, serialization, and expected graph behavior.
- `repair_agent` minimally patches code when deterministic checks fail.

Each role is selected with `src/server.py --role <role>`. When Amber wires the role slots, the
coordinator delegates over A2A using `PLANNER_AGENT_URL`, `CODER_AGENT_URL`, and
`REPAIR_AGENT_URL`. Without those slots, the same container still runs as a standalone coordinator
and calls the configured LLM for each role locally.

LLM access is intentionally isolated in `src/llm.py`. Configure it with `MODEL_NAME` plus the
`LITELLM_*` variables, or bind an Amber `llm` slot so the component receives `LLM_API_URL`.
If no model is configured, the agent returns a deterministic `process_graph(graph_data)` fallback
so A2A conformance checks and Docker startup still work.
