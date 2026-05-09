from __future__ import annotations

from pathlib import Path


def read_questions(questions_path: str | Path) -> list[str]:
    content = Path(questions_path).read_text(encoding="utf-8")
    questions = [block.strip() for block in content.split("\n\n") if block.strip()]

    normalized: list[str] = []
    for question in questions:
        lines = question.splitlines()
        if len(lines) > 1:
            normalized.append("\n".join(lines[1:]).strip())
        else:
            normalized.append(question)
    return normalized
