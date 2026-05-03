from __future__ import annotations

from config import AgentConfig


class LLMClient:
    def __init__(self, config: AgentConfig):
        self.config = config

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str | None:
        if not self.config.has_llm:
            return None

        try:
            import litellm
            from litellm import CustomStreamWrapper, ModelResponse, ModelResponseStream, acompletion
        except ImportError:
            return None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await acompletion(
            model=self.config.model_name,
            messages=messages,
            api_key=self.config.litellm_api_key,
            base_url=self.config.litellm_api_base_url or self.config.llm_api_url,
            api_version=self.config.litellm_api_version,
            stream=True,
        )

        if isinstance(response, CustomStreamWrapper):
            chunks = [chunk async for chunk in response]
            response = litellm.stream_chunk_builder(chunks, messages=messages)

        if isinstance(response, ModelResponseStream):
            return response.choices[0].delta.content or ""
        if isinstance(response, ModelResponse):
            return response.choices[0].message.content or ""
       