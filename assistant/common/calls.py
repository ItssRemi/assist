import logging
import typing as t
from typing import List, Optional

import httpx
import openai
from aiocache import cached
from openai.types import CreateEmbeddingResponse, Image, ImagesResponse
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from sentry_sdk import add_breadcrumb
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from .constants import PRICES, SUPPORTS_SEED, NO_DEVELOPER_ROLE, SUPPORTS_TOOLS

SUPPORTS_SEED_SET   = set(SUPPORTS_SEED)
NO_DEV_SET          = set(NO_DEVELOPER_ROLE)
SUPPORTS_TOOLS_SET  = set(SUPPORTS_TOOLS)

log = logging.getLogger("red.vrt.assistant.calls")

@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ReadTimeout, openai.InternalServerError)),
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def request_chat_completion_raw(
    model: str,
    messages: List[dict],
    temperature: float,
    api_key: str,
    max_tokens: Optional[int] = None,             # correct param for chat.completions
    functions: Optional[List[dict]] = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: int | None = None,
    base_url: Optional[str] = None,
    reasoning_effort: Optional[str] = None,       # 'low'|'medium'|'high' (mapped to reasoning.effort)
    verbosity: Optional[str] = None,              # 'low'|'medium'|'high' (sent via extra_body)
) -> ChatCompletion:
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    kwargs: dict = {"model": model, "messages": messages}

    if max_tokens and max_tokens > 0:
        kwargs["max_tokens"] = max_tokens

    # Only apply classic chat knobs to priced OpenAI models that accept them
    if model in PRICES and not (model.startswith("o") or "gpt-5" in model or model.startswith("gpt-4.1")):
        kwargs["temperature"] = temperature
        kwargs["frequency_penalty"] = frequency_penalty
        kwargs["presence_penalty"] = presence_penalty

    # Seed (if supported)
    if seed and (model in SUPPORTS_SEED_SET):
        kwargs["seed"] = seed

    # Reasoning/verbosity wiring for newer families
    if (model.startswith(("o", "gpt-4.1")) or "gpt-5" in model):
        if reasoning_effort:
            eff = "low" if reasoning_effort == "minimal" else reasoning_effort
            kwargs["reasoning"] = {"effort": eff}
        if verbosity:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "verbosity": verbosity}

    # Tools vs legacy functions (prefer tools for modern models if supported)
    if functions and (model not in NO_DEV_SET):
        if model in SUPPORTS_TOOLS_SET:
            kwargs["tools"] = [{"type": "function", "function": f} for f in functions]
        else:
            # Legacy (rarely needed now, but kept for compatibility)
            kwargs["functions"] = functions

    # Call
    try:
        resp: ChatCompletion = await client.chat.completions.create(**kwargs)
    except TypeError:
        # Defensive retry with experimental fields stripped
        kwargs.pop("extra_body", None)
        kwargs.pop("reasoning", None)
        resp = await client.chat.completions.create(**kwargs)

    log.debug("request_chat_completion_raw: %s -> %s", model, resp.model)
    return resp


@cached(ttl=60)
@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ReadTimeout, openai.InternalServerError)),
    wait=wait_random_exponential(min=5, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def request_embedding_raw(
    text: str,
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
) -> CreateEmbeddingResponse:
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    add_breadcrumb(category="api", message="Calling request_embedding_raw", level="info", data={"text": text})
    response: CreateEmbeddingResponse = await client.embeddings.create(input=text, model=model)
    log.debug("request_embedding_raw: %s -> %s", model, response.model)
    return response


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ReadTimeout, openai.InternalServerError)),
    wait=wait_random_exponential(min=5, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def request_image_raw(
    prompt: str,
    api_key: str,
    size: t.Literal["1024x1024", "1792x1024", "1024x1792", "1024x1536", "1536x1024"] = "1024x1024",
    quality: t.Literal["standard", "hd", "low", "medium", "high"] = "standard",
    style: t.Optional[t.Literal["natural", "vivid"]] = "vivid",
    model: t.Literal["dall-e-3", "gpt-image-1"] = "dall-e-3",
    base_url: Optional[str] = None,
) -> Image:
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    kwargs = {"model": model, "prompt": prompt, "size": size, "n": 1}

    if model == "dall-e-3":
        kwargs["quality"] = quality if quality in ["standard", "hd"] else "standard"
        kwargs["style"] = style
        kwargs["response_format"] = "b64_json"
    elif model == "gpt-image-1":
        kwargs["quality"] = quality if quality in ["low", "medium", "high"] else "medium"

    response: ImagesResponse = await client.images.generate(**kwargs)
    return response.data[0]


async def request_image_edit_raw(
    prompt: str,
    api_key: str,
    images: t.List[bytes],  # bytes for each image
    base_url: Optional[str] = None,
) -> Image:
    assert all(isinstance(image, bytes) for image in images), "All images must be bytes."
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    response: ImagesResponse = await client.images.edit(
        model="gpt-image-1",
        prompt=prompt,
        image=images,
    )
    return response.data[0]


class CreateMemoryResponse(BaseModel):
    memory_name: str
    memory_content: str


async def create_memory_call(
    messages: t.List[dict],
    api_key: str,
    base_url: Optional[str] = None,
) -> t.Union[CreateMemoryResponse, None]:
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    response = await client.beta.chat.completions.parse(
        model="o3",
        messages=messages,
        response_format=CreateMemoryResponse,
    )
    return response.choices[0].message.parsed
