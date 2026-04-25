from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
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
        """Run the role-specific agent loop.

        The coordinator can delegate to planner/verifier A2A helpers when Amber
        wires those slots. Each role can also operate as a standalone A2A agent.
        """
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"{self.role.name} is preparing a response..."),
        )

        result = await self._respond(input_text)

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=result))],
            name=f"{self.role.name}-response",
        )

    async def _respond(self, input_text: str) -> str:
        if self.role.name == "coordinator":
            return await self._coordinate(input_text)
        return await self._role_response(input_text)

    async def _coordinate(self, input_text: str) -> str:
        plan = await self._delegate(
            self.config.planner_agent_url,
            f"Plan a Kubernetes benchmark response for:\n\n{input_text}",
        )

        draft_prompt = self._build_prompt(input_text, plan=plan)
        draft = await self.llm.complete(
            system_prompt=self.role.system_prompt,
            user_prompt=draft_prompt,
        )
        if not draft:
            draft = self._fallback_response(input_text, plan=plan)

        verification = await self._delegate(
            self.config.verifier_agent_url,
            f"Verify this Kubernetes benchmark response:\n\n{draft}",
        )
        if verification:
            return f"{draft}\n\nVerification notes:\n{verification}"
        return draft

    async def _role_response(self, input_text: str) -> str:
        response = await self.llm.complete(
            system_prompt=self.role.system_prompt,
            user_prompt=input_text,
        )
        return response or self._fallback_response(input_text)

    async def _delegate(self, url: str | None, message: str) -> str | None:
        if not url:
            return None
        try:
            return await self.messenger.talk_to_agent(message, url)
        except Exception as exc:
            return f"Delegation to {url} failed: {exc}"

    def _build_prompt(self, input_text: str, plan: str | None = None) -> str:
        sections = [f"User request:\n{input_text}"]
        if plan:
            sections.append(f"Planner notes:\n{plan}")
        return "\n\n".join(sections)

    def _fallback_response(self, input_text: str, plan: str | None = None) -> str:
        lines = [
            f"Role: {self.role.name}",
            "Status: scaffold response; configure MODEL_NAME and LITELLM_* or bind an Amber llm slot for live reasoning.",
            "",
            "K8s response skeleton:",
            "1. Restate the objective and the namespace/cluster assumptions.",
            "2. Inspect relevant pods, services, events, RBAC, network policies, and logs.",
            "3. Propose the smallest reversible action that advances the benchmark goal.",
            "4. Verify with concrete cluster evidence before reporting completion.",
            "",
            "Input:",
            input_text,
        ]
        if plan:
            lines.extend(["", "Planner notes:", plan])
        return "\n".join(lines)
