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
    proposer_agent_url: str | None = None
    reviewer_agent_url: str | None = None
    verifier_agent_url: str | None = None

    @classmethod
    def from_env(
        cls,
        role: str | None = None,
        model_name: str | None = None,
        litellm_api_key: str | None = None,
        litellm_api_base_url: str | None = None,
        litellm_api_version: str | None = None,
    ) -> "AgentConfig":
        return cls(
            role=role or os.getenv("AGENT_ROLE", "coordinator"),
            model_name=model_name or os.getenv("MODEL_NAME"),
            llm_api_url=os.getenv("LLM_API_URL"),
            litellm_api_key=(
                litellm_api_key
                or os.getenv("LITELLM_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("AZURE_API_KEY")
            ),
            litellm_api_base_url=(
                litellm_api_base_url
                or os.getenv("LITELLM_API_BASE_URL")
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("AZURE_API_BASE")
            ),
            litellm_api_version=(
                litellm_api_version
                or os.getenv("LITELLM_API_VERSION")
                or os.getenv("OPENAI_API_VERSION")
                or os.getenv("AZURE_API_VERSION")
            ),
            planner_agent_url=os.getenv("PLANNER_AGENT_URL"),
            proposer_agent_url=os.getenv("PROPOSER_AGENT_URL"),
            reviewer_agent_url=os.getenv("REVIEWER_AGENT_URL") or os.getenv("VERIFIER_AGENT_URL"),
            verifier_agent_url=os.getenv("VERIFIER_AGENT_URL"),
        )

    @property
    def has_llm(self) -> bool:
        return bool(self.model_name)
