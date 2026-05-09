from __future__ import annotations

import json
import os
import re
from pathlib import Path


def load_or_build_drug_dict(txt_file_path: str | Path, cache_json_path: str | Path) -> dict[str, str]:
    txt_path = Path(txt_file_path)
    cache_path = Path(cache_json_path)

    if cache_path.exists() and cache_path.stat().st_mtime >= txt_path.stat().st_mtime:
        with cache_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    drug_dict: dict[str, str] = {}
    current_drug: str | None = None
    current_info: list[str] = []

    with txt_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = re.sub(r"\\s*", "", line.strip())
            if not line:
                continue
            if line.startswith("-"):
                if current_drug:
                    current_info.append(line)
            else:
                if current_drug and current_info:
                    drug_dict[current_drug] = "\n".join(current_info)
                current_drug = line
                current_info = []

    if current_drug and current_info:
        drug_dict[current_drug] = "\n".join(current_info)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(drug_dict, file, ensure_ascii=False, indent=2)

    return drug_dict
