from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ipf_rag.core.config import AppConfig, DEFAULT_DEEPSEEK_V3
from ipf_rag.core.prompts import IPF_RAG_TEMPLATE
from ipf_rag.pipelines.rag_pipeline import IPFRAGPipeline
from ipf_rag.utils.io import append_json_record, load_json
from ipf_rag.utils.questions import read_questions


@dataclass(slots=True)
class GenerationTask:
    name: str


def load_existing_records(output_path: str | Path) -> set[tuple[int, str]]:
    data = load_json(output_path, default=[])
    return {(item["question_idx"], item.get("RAG", "Unknown")) for item in data}


def generate_answers(project_root: str | Path, output_path: str | Path, start_question_idx: int = 1, top_k: int = 1) -> None:
    app_config = AppConfig.from_project_root(project_root, default_model=DEFAULT_DEEPSEEK_V3)
    pipeline = IPFRAGPipeline(app_config, DEFAULT_DEEPSEEK_V3)
    questions = read_questions(app_config.data.questions_file)
    existing = load_existing_records(output_path)

    tasks = [GenerationTask(name="IPF-RAG")]

    for idx, question in enumerate(questions, start=1):
        if idx < start_question_idx:
            continue
        for task in tasks:
            if (idx, task.name) in existing:
                continue
            answer = pipeline.generate(question, top_k=top_k, prompt_template=IPF_RAG_TEMPLATE)
            record = {
                "question_idx": idx,
                "question": question,
                "answer": answer,
                "model": DEFAULT_DEEPSEEK_V3.model_name,
                "RAG": task.name,
            }
            append_json_record(output_path, record)
            existing.add((idx, task.name))
