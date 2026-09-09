import os

import dotenv

from loop import agent_step, _backfill_orphan_tool_results
from context import build_context
from providers.anthropic_provider import AnthropicProvider
from providers.base import Provider
from providers.fallback_provider import FallbackProvider
from providers.openai_compatible_provider import OpenAIProvider

import queue
import sys
import threading

dotenv.load_dotenv()

chain: list[Provider] = []
if os.environ.get("ANTHROPIC_API_KEY"):
    chain.append(AnthropicProvider())
if os.environ.get("OPENAI_API_KEY"):
    chain.append(OpenAIProvider(model="gpt-5"))
if os.environ.get("GEMINI_API_KEY"):
    chain.append(OpenAIProvider(
        model="gemini-2.5-flash-lite",
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ))
DEFAULT_PROVIDER = FallbackProvider(chain)


_stdin_q: queue.Queue[str | None] = queue.Queue()


def _stdin_reader() -> None:
    for line in sys.stdin:
        _stdin_q.put(line.rstrip("\n"))
    _stdin_q.put(None)


def _start_stdin_reader_once() -> None:
    if getattr(_start_stdin_reader_once, "_started", False):
        return
    threading.Thread(target=_stdin_reader, daemon=True, name="stdin-reader").start()
    _start_stdin_reader_once._started = True  # type: ignore[attr-defined]

def _next_user_line(prompt: str) -> str | None:
    print(prompt, end="", flush=True)
    while True:
        try:
            return _stdin_q.get(timeout=0.5)
        except queue.Empty:
            continue


def _drain_user_lines() -> list[str]:
    lines: list[str] = []
    while True:
        try:
            item = _stdin_q.get_nowait()
        except queue.Empty:
            break
        if item is None:
            continue
        if item.strip():
            lines.append(item.strip())
    return lines


def chat(provider: Provider | None = None) -> None:
    """Run an interactive chat loop, dispatching each user turn to the agent loop."""
    if provider is None:
        provider = DEFAULT_PROVIDER

    _start_stdin_reader_once()  

    system = build_context()
    messages: list[dict] = []
    print("chat — Ctrl-D or empty line to exit\n")
    while True:
        line = _next_user_line("you: ")
        if line is None:                                    # EOF: stdin closed
            print()
            break
        user_input = line.strip()
        if not user_input:
            break

        print("\nassistant: ", end="", flush=True)
        try:
            agent_step(
                user_input, messages, provider, system,
                on_text_delta=lambda delta: print(delta, end="", flush=True),
                get_injections=_drain_user_lines,
            )
        except Exception as exc:
            print(f"\n\n[error: {type(exc).__name__}: {exc}]")
            _backfill_orphan_tool_results(messages)
            messages.append({
                "role": "assistant",
                "content": f"[error: {type(exc).__name__}]",
            })
        print("\n")


if __name__ == "__main__":
    chat()
