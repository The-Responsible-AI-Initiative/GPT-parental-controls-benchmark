from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence

from .config import ProviderConfig
from .models import ChatMessage


class ProviderError(RuntimeError):
    pass


class TextProvider(Protocol):
    def complete(
        self, messages: Sequence[ChatMessage], *, json_output: bool = False
    ) -> str:
        ...


def _decode_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except OSError:
        body = ""
    return f"HTTP {exc.code}: {body[:1000]}"


class _HTTPProvider:
    retry_statuses = {408, 409, 429, 500, 502, 503, 504}

    def __init__(self, config: ProviderConfig):
        self.config = config

    def _api_key(self) -> str:
        value = os.environ.get(self.config.api_key_env, "").strip()
        if not value:
            raise ProviderError(
                f"required environment variable {self.config.api_key_env!r} is not set"
            )
        return value

    def _post(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        request = urllib.request.Request(
            self.config.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    raw = response.read().decode("utf-8")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ProviderError("provider returned a non-object JSON response")
                return value
            except urllib.error.HTTPError as exc:
                if (
                    exc.code not in self.retry_statuses
                    or attempt == self.config.max_retries
                ):
                    raise ProviderError(_decode_http_error(exc)) from exc
            except urllib.error.URLError as exc:
                if attempt == self.config.max_retries:
                    raise ProviderError(f"provider request failed: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise ProviderError("provider returned invalid JSON") from exc
            time.sleep(min(2 ** attempt, 8))
        raise AssertionError("unreachable retry state")


class OpenAICompatibleProvider(_HTTPProvider):
    def complete(
        self, messages: Sequence[ChatMessage], *, json_output: bool = False
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        response = self._post(
            payload, {"Authorization": f"Bearer {self._api_key()}"}
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "OpenAI-compatible response lacks choices[0].message.content"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("OpenAI-compatible provider returned empty content")
        return content


class AnthropicProvider(_HTTPProvider):
    def complete(
        self, messages: Sequence[ChatMessage], *, json_output: bool = False
    ) -> str:
        system_parts = [
            message.content for message in messages if message.role == "system"
        ]
        conversation = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role in {"user", "assistant"}
        ]
        if not conversation:
            raise ProviderError("Anthropic request requires at least one user message")
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": 2048,
            "messages": conversation,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if json_output:
            payload["system"] = (
                str(payload.get("system", ""))
                + "\n\nReturn exactly one valid JSON object and no surrounding prose."
            ).strip()
        response = self._post(
            payload,
            {
                "x-api-key": self._api_key(),
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            blocks = response["content"]
        except KeyError as exc:
            raise ProviderError("Anthropic response lacks content") from exc
        if not isinstance(blocks, list):
            raise ProviderError("Anthropic response content is not a list")
        text = "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        if not text:
            raise ProviderError("Anthropic provider returned no text blocks")
        return text


class ScriptedProvider:
    def __init__(self, responses_path: Path):
        try:
            value = json.loads(responses_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProviderError(f"scripted responses not found: {responses_path}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"invalid scripted response JSON in {responses_path}: {exc}"
            ) from exc
        if not isinstance(value, list) or not value:
            raise ProviderError("scripted response file must contain a non-empty array")
        self._responses: List[str] = []
        for index, item in enumerate(value):
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text = str(item["text"])
            else:
                raise ProviderError(
                    f"scripted response {index} must be a string or object with text"
                )
            self._responses.append(text)
        self._cursor = 0

    def complete(
        self, messages: Sequence[ChatMessage], *, json_output: bool = False
    ) -> str:
        del messages, json_output
        if self._cursor >= len(self._responses):
            raise ProviderError("scripted provider exhausted its responses")
        response = self._responses[self._cursor]
        self._cursor += 1
        return response


def create_provider(config: ProviderConfig) -> TextProvider:
    if config.kind != "scripted" and config.model == "CHANGE_ME":
        raise ProviderError(
            "configure a provider model before making requests"
        )
    if config.kind == "openai_compatible":
        return OpenAICompatibleProvider(config)
    if config.kind == "anthropic":
        return AnthropicProvider(config)
    if config.kind == "scripted":
        assert config.responses_path is not None
        return ScriptedProvider(config.responses_path)
    raise ProviderError(f"unsupported provider kind: {config.kind}")
