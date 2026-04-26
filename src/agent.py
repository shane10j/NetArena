from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState
from a2a.utils import get_message_text, new_agent_text_message

from config import AgentConfig
from llm import LLMClient
from messenger import Messenger
from roles import get_role


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
        if self.role.name == "coordinator":
            return await self._coordinate(input_text)
        return await self._role_response(input_text)

    async def _coordinate(self, input_text: str) -> str:
        plan = await self._delegate(
            self.config.planner_agent_url,
            f"Plan a safe MALT NetworkX graph solution for this task:\n\n{input_text}",
        )

        draft = await self._draft_response(input_text, plan=plan)
        verification = await self._delegate(
            self.config.verifier_agent_url,
            self._build_verification_prompt(input_text, draft),
        )

        if self._should_revise(verification):
            revised = await self._draft_response(
                input_text,
                plan=plan,
                verification=verification,
                previous=draft,
            )
            if revised:
                return self._normalize_response(revised)

        return self._normalize_response(draft)

    async def _role_response(self, input_text: str) -> str:
        response = await self.llm.complete(
            system_prompt=self.role.system_prompt,
            user_prompt=input_text,
        )
        return self._normalize_response(response or self._fallback_response(input_text))

    async def _draft_response(
        self,
        input_text: str,
        *,
        plan: str | None = None,
        verification: str | None = None,
        previous: str | None = None,
    ) -> str:
        print(f"Calling LiteLLM model: {self.config.model_name or '<none configured>'}")
        response = await self.llm.complete(
            system_prompt=self.role.system_prompt,
            user_prompt=self._build_draft_prompt(
                input_text,
                plan=plan,
                verification=verification,
                previous=previous,
            ),
        )
        return response or self._fallback_response(input_text)

    async def _delegate(self, url: str | None, message: str) -> str | None:
        if not url:
            return None
        try:
            return await self.messenger.talk_to_agent(message, url)
        except Exception as exc:
            return f"Delegation to {url} failed: {exc}"

    def _build_draft_prompt(
        self,
        input_text: str,
        *,
        plan: str | None = None,
        verification: str | None = None,
        previous: str | None = None,
    ) -> str:
        sections = [
            "Generate the Python implementation for the benchmark prompt below.",
            "Use only normal NetworkX graph operations and the provided graph_data object.",
            f"Benchmark prompt:\n{input_text}",
        ]
        if plan:
            sections.append(f"Planner notes:\n{plan}")
        if previous and verification:
            sections.append(f"Previous draft:\n{previous}")
            sections.append(f"Verifier feedback:\n{verification}")
            sections.append("Revise the code to address the verifier feedback.")
        return "\n\n".join(sections)

    def _build_verification_prompt(self, input_text: str, draft: str) -> str:
        return "\n\n".join(
            [
                "Review this MALT NetworkX benchmark answer for executable Python, correctness, and safety.",
                "Check that it uses process_graph(graph_data), copies the graph before mutation, preserves unrelated attributes,",
                "returns type/data/updated_graph when possible, and avoids benchmark-private helper functions.",
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

    def _fallback_response(self, input_text: str) -> str:
        return "\n".join(
            [
                "def process_graph(graph_data):",
                "    graph_copy = graph_data.copy()",
                "    graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)",
                "    return {'type': 'graph', 'data': graph_json, 'updated_graph': graph_json}",
            ]
        )
