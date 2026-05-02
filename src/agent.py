from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState
from a2a.utils import get_message_text, new_agent_text_message

from config import AgentConfig
from llm import LLMClient
from messenger import Messenger
from roles import get_role


PRIVATE_HELPER_MARKERS = (
    "solid_step_",
    "malt_step_",
)


class Agent:
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig.from_env()
        self.messenger = Messenger()
        self.llm = LLMClient(self.config)
        self.role = get_role(self.config.role)

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"{self.role.name} is preparing a response..."),
        )

        result = await self._respond(input_text)
        await updater.complete(new_agent_text_message(result))

    async def invoke(self, input_text: str) -> str:
        return await self._respond(input_text)

    async def _respond(self, input_text: str) -> str:
        if self._is_conformance_ping(input_text):
            return "Agent is ready."
        if self.role.name == "coordinator":
            return await self._coordinate(input_text)
        return await self._role_response(input_text)

    async def _coordinate(self, input_text: str) -> str:
        plan = await self._delegate(
            self.config.planner_agent_url,
            f"Plan a safe MALT NetworkX graph solution for this task:\n\n{input_text}",
        )

        draft = await self._propose_response(input_text, plan=plan)
        feedback = await self._review_response(input_text, draft)

        if self._should_revise(feedback):
            revised = await self._propose_response(
                input_text,
                plan=plan,
                feedback=feedback,
                previous=draft,
            )
            final_feedback = await self._review_response(input_text, revised)
            if not self._should_revise(final_feedback):
                return self._normalize_response(revised)
            return self._fallback_response(input_text)

        return self._normalize_response(draft)

    async def _propose_response(
        self,
        input_text: str,
        *,
        plan: str | None = None,
        feedback: str | None = None,
        previous: str | None = None,
    ) -> str:
        prompt = self._build_proposal_prompt(
            input_text,
            plan=plan,
            feedback=feedback,
            previous=previous,
        )
        delegated = await self._delegate(self.config.proposer_agent_url, prompt)
        if delegated:
            return delegated

        return await self._complete_with_role("proposer", prompt, input_text)

    async def _review_response(self, input_text: str, draft: str) -> str:
        prompt = self._build_review_prompt(input_text, draft)
        delegated = await self._delegate(
            self.config.reviewer_agent_url or self.config.verifier_agent_url,
            prompt,
        )
        remote_feedback = delegated
        if remote_feedback is None and self.config.has_llm:
            remote_feedback = await self._complete_with_role("reviewer", prompt, input_text)
        local_feedback = self._local_review(draft)

        issues = [
            feedback
            for feedback in (remote_feedback, local_feedback)
            if feedback and feedback.strip().lower() != "pass"
        ]
        if issues:
            return "\n".join(issues)
        return "PASS"

    async def _role_response(self, input_text: str) -> str:
        response = await self.llm.complete(
            system_prompt=self.role.system_prompt,
            user_prompt=input_text,
        )
        if not response and self.role.name in {"reviewer", "verifier"}:
            return self._local_review(self._extract_review_draft(input_text))
        return self._normalize_response(response or self._fallback_response(input_text))

    async def _complete_with_role(self, role: str, prompt: str, input_text: str) -> str:
        response = await self.llm.complete(
            system_prompt=get_role(role).system_prompt,
            user_prompt=prompt,
        )
        return response or self._fallback_response(input_text)

    async def _delegate(self, url: str | None, message: str) -> str | None:
        if not url:
            return None
        try:
            return await self.messenger.talk_to_agent(message, url)
        except Exception as exc:
            return f"Delegation to {url} failed: {exc}"

    def _build_proposal_prompt(
        self,
        input_text: str,
        *,
        plan: str | None = None,
        feedback: str | None = None,
        previous: str | None = None,
    ) -> str:
        sections = [
            "Generate a pure NetworkX Python implementation for the benchmark prompt below.",
            "Rules:",
            "- Return only executable Python code with no Markdown fences.",
            "- Define process_graph(graph_data).",
            "- Use graph_data.copy() before mutating.",
            "- Use only normal NetworkX and Python APIs.",
            "- Do not call solid_step_* or any benchmark-private helper.",
            "- Mimic helper behavior explicitly with node attribute matching and graph traversal.",
            f"Benchmark prompt:\n{input_text}",
        ]
        if plan:
            sections.append(f"Planner notes:\n{plan}")
        if previous and feedback:
            sections.append(f"Previous draft:\n{previous}")
            sections.append(f"Reviewer feedback:\n{feedback}")
            sections.append("Revise the code to satisfy every rule and reviewer issue.")
        return "\n\n".join(sections)

    def _build_review_prompt(self, input_text: str, draft: str) -> str:
        return "\n\n".join(
            [
                "Review this MALT NetworkX benchmark answer for executable Python, correctness, and safety.",
                "Check that it uses only pure NetworkX/Python, defines process_graph(graph_data), copies before mutation,",
                "preserves unrelated attributes, returns type/data/updated_graph when possible, and avoids benchmark-private helpers.",
                "Reject any call or reference to solid_step_* or other private helper behavior; require explicit NetworkX logic instead.",
                f"Benchmark prompt:\n{input_text}",
                f"Draft answer:\n{draft}",
                "Reply with PASS if acceptable, otherwise list concise issues to fix.",
            ]
        )

    def _should_revise(self, verification: str | None) -> bool:
        if not verification:
            return False
        text = verification.strip().lower()
        return bool(text) and not text.startswith("pass")

    def _normalize_response(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("Answer:"):
            stripped = stripped[len("Answer:") :].strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        return stripped

    def _local_review(self, text: str) -> str:
        stripped = self._normalize_response(text)
        issues = []
        if any(marker in stripped for marker in PRIVATE_HELPER_MARKERS):
            issues.append("Do not call or reference benchmark-private helper functions; implement the behavior with pure NetworkX.")
        if "def process_graph(graph_data)" not in stripped:
            issues.append("Define process_graph(graph_data).")
        if "```" in text:
            issues.append("Return raw Python only, without Markdown fences.")
        if "graph_data.copy()" not in stripped and ".copy()" not in stripped:
            issues.append("Copy graph_data before mutating or serializing the graph.")
        if issues:
            return "\n".join(issues)
        return "PASS"

    def _extract_review_draft(self, text: str) -> str:
        marker = "Draft answer:\n"
        if marker not in text:
            return text
        draft = text.split(marker, 1)[1]
        return draft.rsplit("\nReply with PASS", 1)[0]

    def _is_conformance_ping(self, text: str) -> bool:
        return text.strip().lower() in {
            "hello",
            "hi",
            "ping",
            "health",
            "health check",
            "status",
        }

    def _fallback_response(self, input_text: str) -> str:
        return "\n".join(
            [
                "def process_graph(graph_data):",
                "    import networkx as nx",
                "    graph_copy = graph_data.copy()",
                "    graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)",
                "    return {'type': 'graph', 'data': graph_json, 'updated_graph': graph_json}",
            ]
        )
