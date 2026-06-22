"""Exercise 3: Bootstrap-file ordering, the nanobot way.

Replace the alphabetical `glob("*.md")` with an explicit ordered list
and prefix each file's content with a `## <filename>` header. The model
can now refer to your instruction files by name.

Probe it with: "which of your instruction files should I edit if I want
to change your tone?" The model will answer `persona.md`, by name —
something the original `build_context()` could not do, because it never
told the model what the files were called.

Compare this to `_load_bootstrap_files` in nanobot/nanobot/agent/context.py.
"""

import os
import pathlib

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-4-6"

TEMPLATES_DIR = pathlib.Path(__file__).resolve().parent.parent / "core" / "templates"
BOOTSTRAP_FILES = ["AGENTS.md", "persona.md", "instructions.md"]


def build_context_ordered() -> str:
    """Load BOOTSTRAP_FILES in order, prefixing each with a `## <filename>` header."""
    parts: list[str] = []
    for name in BOOTSTRAP_FILES:
        path = TEMPLATES_DIR / name
        if path.exists():
            parts.append(f"## {name}\n\n{path.read_text()}")
    return "\n\n".join(parts)


def build_context_glob() -> str:
    """The original Chapter 3 version, for comparison."""
    parts = [p.read_text() for p in sorted(TEMPLATES_DIR.glob("*.md"))]
    return "\n\n".join(parts)


def ask(system: str, prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    probe = "Which of your instruction files should I edit if I want to change your tone?"

    print("=== glob version (no filenames in prompt) ===")
    print(ask(build_context_glob(), probe))
    print()
    print("=== ordered version with `## <filename>` headers ===")
    print(ask(build_context_ordered(), probe))
