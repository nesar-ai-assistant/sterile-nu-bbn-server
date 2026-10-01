"""Server-side skills: named recipes for using this server's tools well.

Each skill is a markdown file in skills/ with a small frontmatter header
(name, description). Clients list name + description first and load the
full text only when a task matches; the same files are also registered as
native MCP prompts.
"""

from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


def parse_frontmatter(path: Path) -> dict:
    meta, lines = {}, path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta


def skill_files() -> list[Path]:
    return sorted(SKILLS_DIR.glob("*.md"))


def skill_index() -> dict[str, str]:
    """{name: description} for every skill on disk."""
    index = {}
    for path in skill_files():
        meta = parse_frontmatter(path)
        if "name" in meta:
            index[meta["name"]] = meta.get("description", "")
    return index


def skill_text(name: str) -> str | None:
    for path in skill_files():
        if parse_frontmatter(path).get("name") == name:
            return path.read_text(encoding="utf-8")
    return None
