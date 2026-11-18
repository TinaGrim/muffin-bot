#!/usr/bin/env python3
"""Switch the bot between the local Ollama model and OpenCode Go.

Examples:
  python use_backend.py local
  python use_backend.py go --model gemini-4-argon
  python use_backend.py go --model claude-haiku-5-5

The OpenCode key is read from ~/.local/share/opencode/auth.json and written to
.env without being printed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".env"
AUTH = Path(os.path.expanduser("~/.local/share/opencode/auth.json"))

GO_BASE = "https://opencode.ai/inference/go/v1"
LOCAL_BASE = "http://localhost:11434/v1"


def set_env(lines: list[str], key: str, value: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(key)}=")
    replaced = False
    out = []
    for line in lines:
        if pattern.match(line):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    return out


def opencode_key() -> str:
    if not AUTH.exists():
        raise SystemExit(f"No OpenCode auth file at {AUTH}. Run `opencode auth login` first.")
    data = json.loads(AUTH.read_text(encoding="utf-8"))
    key = (data.get("opencode") or {}).get("key")
    if not key:
        raise SystemExit("No OpenCode API key found in auth.json.")
    return key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["local", "go"])
    parser.add_argument("--model", default="gemini-4-argon")
    args = parser.parse_args()

    if not ENV.exists():
        raise SystemExit("No .env file. Copy .env.example to .env first.")
    lines = ENV.read_text(encoding="utf-8").splitlines()

    if args.backend == "local":
        lines = set_env(lines, "LLM_BASE_URL", LOCAL_BASE)
        lines = set_env(lines, "LLM_API_KEY", "ollama")
        lines = set_env(lines, "LLM_MODEL", "qwen2.5:7b-instruct")
        lines = set_env(lines, "EXAMPLES_MODE", "always")
        label = "local Qwen2.5 7B (Ollama)"
    else:
        lines = set_env(lines, "LLM_BASE_URL", GO_BASE)
        lines = set_env(lines, "LLM_API_KEY", opencode_key())
        lines = set_env(lines, "LLM_MODEL", args.model)
        # Remote reasoning models do better without copied examples.
        lines = set_env(lines, "EXAMPLES_MODE", "never")
        # A different model on the same account still hits the same Go usage
        # cap, so the real fallback is local Ollama.
        lines = set_env(lines, "FALLBACK_BASE_URL", "http://localhost:11434/v1")
        lines = set_env(lines, "FALLBACK_API_KEY", "ollama")
        lines = set_env(lines, "FALLBACK_MODEL", "qwen2.5:3b-instruct")
        label = f"OpenCode Go · {args.model}"

    lines = set_env(lines, "ADAPTER_MODE", "false")
    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"backend -> {label}")
    print("restart the bot: kill the running bot.py, then ./run.sh")
    print("revert: python use_backend.py local")


if __name__ == "__main__":
    main()
