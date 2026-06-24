#!/usr/bin/env python3
"""Validate configured LLM clients with minimal safe calls.

This script loads `.env` through the provider layer, but never prints API keys
or environment variable values. It reports only whether each configured model
can complete a tiny request.
"""

import argparse
import base64
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_clients import build_client  # noqa: E402
from src.utils.token_usage import TokenUsageTracker  # noqa: E402


CLIENT_ORDER = [
    ("kimi_client", "student_text_extraction_Kimi"),
    ("deepseek_client", "student_text_extraction_DeepSeek"),
    ("minimax_client", "student_text_extraction_MiniMax"),
    ("mimo_vision_client", "table_vision_extraction_MiMo"),
    ("aggregation_client", "teacher_aggregation_configured"),
    ("review_client", "review_agent"),
]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


FALLBACK_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAATElEQVR4nO3P"
    "MQEAAAgDINc/9F3hA2QK0JkzAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAADwG0wAAQABJzQAAAAASUVORK5CYII="
)


def write_healthcheck_image(path: str) -> None:
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (64, 64), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle((8, 8, 56, 56), outline="black", width=2)
        draw.text((22, 24), "OK", fill="black")
        img.save(path, format="PNG")
    except Exception:
        with open(path, "wb") as f:
            f.write(base64.b64decode(FALLBACK_PNG_B64))


def validate_one(name: str, stage: str, cfg: Dict[str, Any], timeout_note: str = "") -> Dict[str, Any]:
    provider = cfg.get("provider")
    model = cfg.get("model_name")
    started = time.time()
    row = {
        "client": name,
        "stage": stage,
        "provider": provider,
        "model_name": model,
        "success": False,
        "latency_seconds": None,
        "output_preview": "",
        "error_message": "",
        "timeout_note": timeout_note,
    }
    if not provider or not model:
        row["error_message"] = "missing provider or model_name in config"
        return row

    try:
        client = build_client(provider, model)
        if "vision" in stage.lower():
            image_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    image_path = tmp.name
                write_healthcheck_image(image_path)
                response = client.chat(
                    messages=[
                        {
                            "role": "user",
                            "text": "This is a health check image. Return only OK.",
                            "image_path": image_path,
                        }
                    ],
                    is_multimodal=True,
                    temperature=0,
                    max_tokens=128,
                    task=f"healthcheck_{stage}",
                )
            finally:
                if image_path:
                    try:
                        os.remove(image_path)
                    except OSError:
                        pass
        else:
            response = client.chat(
                messages=[
                    {"role": "system", "content": "Return only the word OK."},
                    {"role": "user", "content": "Health check."},
                ],
                temperature=0,
                max_tokens=128,
                task=f"healthcheck_{stage}",
            )
        row["success"] = bool(str(response or "").strip())
        row["output_preview"] = str(response or "").strip()[:80]
    except Exception as exc:
        row["error_message"] = str(exc)
    finally:
        row["latency_seconds"] = round(time.time() - started, 3)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate configured MycoExtract LLM models.")
    parser.add_argument("--config", default="config/extraction_config_v8.yaml")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    TokenUsageTracker.reset()
    config = load_config(Path(args.config))
    clients = config.get("llm_clients", {})

    results: List[Dict[str, Any]] = []
    for name, stage in CLIENT_ORDER:
        if name not in clients:
            continue
        cfg = clients[name] or {}
        results.append(validate_one(name, stage, cfg))

    payload = {
        "config": args.config,
        "all_success": all(r["success"] for r in results),
        "results": results,
        "token_usage": TokenUsageTracker.summary(),
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "config": payload["config"],
        "all_success": payload["all_success"],
        "results": [
            {
                "client": r["client"],
                "provider": r["provider"],
                "model_name": r["model_name"],
                "success": r["success"],
                "latency_seconds": r["latency_seconds"],
                "error_message": r["error_message"],
            }
            for r in results
        ],
    }, ensure_ascii=False, indent=2))

    return 0 if payload["all_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
