import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    role: str = "coordinator"
    model_name: str | None = None
    llm_api_url: str | None = None
    litellm_api_key: str | None = None
    litellm_api_base_url: str | None = None
    litellm_api_version: str | None = None
    planner_agent_url: str | None = None
    verifier_agent_url: str | None = None

    @classmethod
    def from_env(cls, role: str | None = None) -> "AgentConfig":
        return cls(
            role=role or os.getenv("AGENT_ROLE", "coordinator"),
            model_name=os.getenv("MODEL_NAME"),
            llm_api_url=os.getenv("LLM_API_URL"),
            litellm_api_key=os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            litellm_api_base_url=os.getenv("LITELLM_API_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            litellm_api_version=os.getenv("LITELLM_API_VERSION"),
            planner_agent_url=os.getenv("PLANNER_AGENT_URL"),
            verifier_agent_url=os.getenv("VERIFIER_AGENT_URL"),
        )

    @property
    def has_llm(self) -> bool:
        return bool(self.model_name)
