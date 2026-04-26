from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    new_agent_text_message,
)

from agent import Agent
from config import AgentConfig


class Executor(AgentExecutor):
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig.from_env()
        self.agent = Agent(self.config)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        try:
            result = await self.agent.invoke(context.get_user_input())
            await event_queue.enqueue_event(new_agent_text_message(result))
        except Exception as e:
            print(f"Task failed with agent error: {e}")
            await event_queue.enqueue_event(new_agent_text_message(f"Agent error: {e}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
