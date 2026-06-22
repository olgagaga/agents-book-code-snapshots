import pathlib

TEMPLATES_DIR = pathlib.Path(__file__).parent / "templates"


def build_context() -> str:
    """Read all Markdown files in the templates directory and concatenate them
    into a single system prompt string."""
    if not TEMPLATES_DIR.exists():
        return ""
    parts: list[str] = []
    for md_file in sorted(TEMPLATES_DIR.glob("*.md")):
        parts.append(md_file.read_text())
    return "\n\n".join(parts)
