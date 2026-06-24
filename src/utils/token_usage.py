"""
Unified token and cost accounting for LLM calls.

The tracker records API-reported usage when a provider returns it. If usage is
missing, it falls back to a conservative offline estimate so every call is still
represented in run summaries and ablation outputs.
"""
from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_PRICING_PER_1M = {
    # Keep these editable in downstream scripts; prices change frequently.
    "openai": {"input": 0.0, "output": 0.0, "currency": "USD"},
    "gpt5client": {"input": 0.0, "output": 0.0, "currency": "USD"},
    "deepseekclient": {"input": 0.0, "output": 0.0, "currency": "USD"},
    "moonshotclient": {"input": 0.0, "output": 0.0, "currency": "CNY"},
    "zhipuaiclient": {"input": 0.4, "output": 0.8, "currency": "CNY"},
    "anthropicclient": {"input": 0.0, "output": 0.0, "currency": "USD"},
    "geminiclient": {"input": 0.0, "output": 0.0, "currency": "USD"},
    "ollamaclient": {"input": 0.0, "output": 0.0, "currency": "LOCAL"},
}


@dataclass
class TokenUsageRecord:
    timestamp: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    request_count: int = 1
    source: str = "api"
    is_multimodal: bool = False
    task: Optional[str] = None


class TokenUsageTracker:
    """Process-wide token tracker shared by all LLM clients."""

    _lock = threading.Lock()
    _records: List[TokenUsageRecord] = []

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._records = []

    @classmethod
    def record(
        cls,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        source: str = "api",
        is_multimodal: bool = False,
        task: Optional[str] = None,
    ) -> None:
        prompt_tokens = max(int(prompt_tokens or 0), 0)
        completion_tokens = max(int(completion_tokens or 0), 0)
        with cls._lock:
            cls._records.append(
                TokenUsageRecord(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    provider=provider,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    source=source,
                    is_multimodal=is_multimodal,
                    task=task,
                )
            )

    @classmethod
    def record_from_response(
        cls,
        provider: str,
        model: str,
        response: Any,
        messages: Iterable[Dict[str, Any]],
        response_text: str,
        is_multimodal: bool = False,
        task: Optional[str] = None,
    ) -> None:
        usage = cls.extract_usage(response)
        if usage:
            source = "api"
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
        else:
            source = "estimated"
            prompt_tokens = cls.estimate_messages(messages, is_multimodal=is_multimodal)
            completion_tokens = cls.estimate_text(response_text)

        cls.record(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            source=source,
            is_multimodal=is_multimodal,
            task=task,
        )

    @staticmethod
    def extract_usage(response: Any) -> Optional[Dict[str, int]]:
        """Read usage from common OpenAI/Anthropic/Gemini/Ollama response shapes."""
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            usage = getattr(response, "usage_metadata", None)

        if usage is None:
            return None

        def pick(*names: str) -> int:
            for name in names:
                if isinstance(usage, dict) and name in usage:
                    return int(usage.get(name) or 0)
                value = getattr(usage, name, None)
                if value is not None:
                    return int(value or 0)
            return 0

        prompt = pick("prompt_tokens", "input_tokens", "prompt_token_count")
        completion = pick("completion_tokens", "output_tokens", "candidates_token_count")
        total = pick("total_tokens", "total_token_count")
        if not prompt and not completion and total:
            return {"prompt_tokens": total, "completion_tokens": 0}
        if not prompt and not completion:
            return None
        return {"prompt_tokens": prompt, "completion_tokens": completion}

    @classmethod
    def estimate_messages(cls, messages: Iterable[Dict[str, Any]], is_multimodal: bool = False) -> int:
        total = 0
        image_count = 0
        for message in messages:
            total += cls.estimate_text(str(message.get("role", "")))
            content = message.get("content") or message.get("text") or ""
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += cls.estimate_text(item.get("text", ""))
                    elif isinstance(item, dict) and "image" in str(item.get("type", "")):
                        image_count += 1
                    else:
                        total += cls.estimate_text(str(item))
            else:
                total += cls.estimate_text(str(content))
            if message.get("image_path"):
                paths = message["image_path"] if isinstance(message["image_path"], list) else [message["image_path"]]
                image_count += len(paths)

        # Vision tokenization is provider-specific. Use a transparent placeholder
        # only when the API does not return real usage.
        if is_multimodal and image_count:
            total += 1024 * image_count
        return total

    @staticmethod
    def estimate_text(text: str) -> int:
        if not text:
            return 0
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # Mixed Chinese/English scientific text: this is intentionally
            # conservative and stable without extra dependencies.
            cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
            non_cjk = len(text) - cjk
            return max(1, cjk + math.ceil(non_cjk / 4))

    @classmethod
    def summary(cls, pricing: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        pricing = pricing or DEFAULT_PRICING_PER_1M
        with cls._lock:
            records = list(cls._records)

        by_model: Dict[str, Dict[str, Any]] = {}
        by_provider: Dict[str, Dict[str, Any]] = {}
        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
            "estimated_request_count": 0,
        }

        for record in records:
            provider_key = record.provider.lower()
            model_key = f"{record.provider}:{record.model}"
            for bucket, key in ((by_model, model_key), (by_provider, provider_key)):
                if key not in bucket:
                    bucket[key] = {
                        "provider": record.provider,
                        "model": record.model if bucket is by_model else None,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "request_count": 0,
                        "estimated_request_count": 0,
                        "multimodal_request_count": 0,
                        "estimated_cost": 0.0,
                        "currency": pricing.get(provider_key, {}).get("currency", ""),
                    }
                item = bucket[key]
                item["prompt_tokens"] += record.prompt_tokens
                item["completion_tokens"] += record.completion_tokens
                item["total_tokens"] += record.total_tokens
                item["request_count"] += 1
                item["estimated_request_count"] += 1 if record.source == "estimated" else 0
                item["multimodal_request_count"] += 1 if record.is_multimodal else 0

            totals["prompt_tokens"] += record.prompt_tokens
            totals["completion_tokens"] += record.completion_tokens
            totals["total_tokens"] += record.total_tokens
            totals["request_count"] += 1
            totals["estimated_request_count"] += 1 if record.source == "estimated" else 0

        for item in by_provider.values():
            price = pricing.get(str(item["provider"]).lower(), {})
            item["estimated_cost"] = (
                item["prompt_tokens"] / 1_000_000 * float(price.get("input", 0.0))
                + item["completion_tokens"] / 1_000_000 * float(price.get("output", 0.0))
            )

        for item in by_model.values():
            price = pricing.get(str(item["provider"]).lower(), {})
            item["estimated_cost"] = (
                item["prompt_tokens"] / 1_000_000 * float(price.get("input", 0.0))
                + item["completion_tokens"] / 1_000_000 * float(price.get("output", 0.0))
            )

        return {
            "totals": totals,
            "by_provider": by_provider,
            "by_model": by_model,
            "records": [asdict(r) for r in records],
        }

    @classmethod
    def save(cls, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(cls.summary(), f, indent=2, ensure_ascii=False)
