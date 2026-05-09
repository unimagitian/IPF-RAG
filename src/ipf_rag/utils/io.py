from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path, default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return [] if default is None else default
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(path: str | Path, data: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def append_json_record(path: str | Path, record: dict) -> None:
    data = load_json(path, default=[])
    data.append(record)
    dump_json(path, data)
