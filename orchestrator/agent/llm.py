"""Which model the agent runs on.

One place, so switching provider is a config change rather than an edit
scattered through the graph. Only an OpenAI key exists on this project today;
the Anthropic branch is here because the demo machine may not be the one this
was written on.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from config import CFG

log = logging.getLogger("orchestrator.llm")


class NoModelConfigured(RuntimeError):
    """Raised at turn time, not import time, so the service still boots."""


@lru_cache(maxsize=4)
def chat_model(model: str | None = None, temperature: float | None = None) -> BaseChatModel:
    """A chat model with tools unbound. Cached per (model, temperature)."""
    name = model or CFG.model

    if not CFG.api_key:
        raise NoModelConfigured(
            f"no API key for provider {CFG.provider!r}. Copy orchestrator/.env.example "
            f"to orchestrator/.env and fill it in."
        )

    if CFG.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise NoModelConfigured(
                "ORCH_PROVIDER=anthropic needs langchain-anthropic installed"
            ) from exc
        return ChatAnthropic(model=name, api_key=CFG.api_key,
                             timeout=CFG.llm_timeout_s, max_retries=1)

    from langchain_openai import ChatOpenAI

    # Some frontier models reject an explicit temperature outright ("only the
    # default (1) is supported"). The frontend hit this and learned to retry;
    # here the simpler answer is not to send one unless asked.
    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature

    # The Responses API, not /v1/chat/completions. A reasoning model refuses
    # the older endpoint outright the moment tools are attached:
    #
    #   400 — Function tools with reasoning_effort are not supported for
    #   gpt-6-astra in /v1/chat/completions. To use function tools, use
    #   /v1/responses or set reasoning_effort to 'none'.
    #
    # Every turn here attaches tools, so the other branch — turning reasoning
    # off — would trade away the thinking that turns a vague sentence into a
    # correct selector. This is the endpoint that keeps both.
    return ChatOpenAI(model=name, api_key=CFG.api_key,
                      timeout=CFG.llm_timeout_s, max_retries=1,
                      use_responses_api=True, **kwargs)


def vision_model() -> BaseChatModel:
    """For event turns, where a snapshot has to be looked at."""
    return chat_model(CFG.vision_model)
