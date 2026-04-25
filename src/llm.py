from __future__ import annotations

from config import AgentConfig


class LLMClient:
    def __init__(self, config: AgentConfig):
        self.config = config

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str | None:
        if not self.config.has_llm:
            return None

        try:
            from litellm import acompletion
        except ImportError:
            return None

        response = await acompletion(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            api_key=self.config.litellm_api_key,
            base_url=self.config.litellm_api_base_url or self.config.llm_api_url,
            api_version=self.config.litellm_api_version,
        )

        choice = response.choices[0]
        return choice.message.content or ""
