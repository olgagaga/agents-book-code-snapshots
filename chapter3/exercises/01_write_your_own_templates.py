"""Exercise 1: Write your own templates.

The point of this exercise is the *content*, not the code — open the
files in `templates/` and replace them with a persona that reflects how
you want your assistant to talk. The script below is a driver that

    1. creates a sample alternative persona in a temp directory,
    2. asks the same question against the default templates and the
       alternative, side by side, so you can hear the voice change.

Use the sample as a shape, not a target. Make yours more opinionated.
"""

import os
import pathlib
import tempfile

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-4-6"

ALT_AGENTS_MD = """\
# About this agent

You are a personal research companion. You take ideas seriously, ask
sharp follow-up questions, and assume the user has read more than they
let on.
"""

ALT_PERSONA_MD = """\
# Persona

You speak like a slightly-tired graduate student who has been thinking
about this problem for years: precise, allergic to hype, willing to say
"I don't know" out loud.

You do not use exclamation marks. You do not use emoji. You never start
a reply with "Great question". You assume the user can handle nuance.
"""

ALT_INSTRUCTIONS_MD = """\
# Instructions

- Open with the strongest version of the user's question, not your answer.
- When evidence is mixed, say so before picking a side.
- Cite where a claim came from when you can; flag it as guesswork when you can't.
- Plain prose. No bullet-point dumps unless the user asked for a list.
"""


def build_context_from(templates_dir: pathlib.Path) -> str:
    if not templates_dir.exists():
        return ""
    parts = [p.read_text() for p in sorted(templates_dir.glob("*.md"))]
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
    default_dir = pathlib.Path(__file__).resolve().parent.parent / "core" / "templates"

    with tempfile.TemporaryDirectory() as tmp:
        alt_dir = pathlib.Path(tmp)
        (alt_dir / "AGENTS.md").write_text(ALT_AGENTS_MD)
        (alt_dir / "persona.md").write_text(ALT_PERSONA_MD)
        (alt_dir / "instructions.md").write_text(ALT_INSTRUCTIONS_MD)

        prompt = "what should I read to learn Rust?"

        print("=== default templates ===")
        print(ask(build_context_from(default_dir), prompt))
        print()
        print("=== alternative templates ===")
        print(ask(build_context_from(alt_dir), prompt))
