"""Exercise 6 (stretch): Declared ordering.

Let `AGENTS.md` declare the load order of the other files. The first
list under the heading `## Load order` inside `AGENTS.md` becomes the
sequence in which `build_context` reads the directory.

Trade-off vs. the hard-coded BOOTSTRAP_FILES list from Exercise 3:
- Hard-coded list:  reordering means editing Python.
- Declared list:    reordering means editing Markdown — but now you
                    have to teach readers a small DSL inside AGENTS.md,
                    and a malformed list silently changes behavior.

To exercise the difference, edit `AGENTS.md` to include:

    ## Load order

    - AGENTS.md
    - persona.md
    - instructions.md

Reorder the bullets and run the script to see the assembled prompt
change without touching Python.
"""

import os
import pathlib
import re

import anthropic
import dotenv

dotenv.load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

TEMPLATES_DIR = pathlib.Path(__file__).resolve().parent.parent / "core" / "templates"
LOAD_ORDER_HEADING = re.compile(r"^##\s+Load order\s*$", re.MULTILINE)
BULLET = re.compile(r"^\s*[-*]\s+(\S+\.md)\s*$", re.MULTILINE)


def parse_load_order(agents_md: str) -> list[str]:
    """Return the list of filenames declared under the first `## Load order` heading.

    If the heading is missing, return an empty list — the caller falls back to
    alphabetical ordering.
    """
    match = LOAD_ORDER_HEADING.search(agents_md)
    if not match:
        return []
    section = agents_md[match.end():]
    next_heading = re.search(r"^##\s", section, re.MULTILINE)
    if next_heading:
        section = section[: next_heading.start()]
    return BULLET.findall(section)


def build_context() -> str:
    if not TEMPLATES_DIR.exists():
        return ""
    agents_path = TEMPLATES_DIR / "AGENTS.md"
    declared = parse_load_order(agents_path.read_text()) if agents_path.exists() else []

    if declared:
        files = [TEMPLATES_DIR / name for name in declared]
    else:
        files = sorted(TEMPLATES_DIR.glob("*.md"))

    return "\n\n".join(f.read_text() for f in files if f.exists())


if __name__ == "__main__":
    agents_md = (TEMPLATES_DIR / "AGENTS.md").read_text()
    declared = parse_load_order(agents_md)
    if declared:
        print(f"declared load order: {declared}")
    else:
        print("no `## Load order` section in AGENTS.md — falling back to alphabetical")
    print()
    print("--- assembled context ---")
    print(build_context())
