from __future__ import annotations

from pathlib import Path

from ipf_rag.core.config import AppConfig, DEFAULT_DEEPSEEK_V3
from ipf_rag.core.prompts import SEM_ONLY_TEMPLATE, STANDARD_RAG_TEMPLATE, TRM_ONLY_TEMPLATE, IPF_RAG_TEMPLATE
from ipf_rag.pipelines.rag_pipeline import IPFRAGPipeline, StandardRAGPipeline
from ipf_rag.utils.io import append_json_record, load_json
from ipf_rag.utils.questions import read_questions


def load_existing_records(output_path: str | Path) -> set[tuple[int, str]]:
    data = load_json(output_path, default=[])
    return {(item["question_idx"], item.get("RAG", "Unknown")) for item in data}


def generate_ablation(project_root: str | Path, output_path: str | Path) -> None:
    app_config = AppConfig.from_project_root(project_root, default_model=DEFAULT_DEEPSEEK_V3)
    ipf_pipeline = IPFRAGPipeline(app_config, DEFAULT_DEEPSEEK_V3)
    std_pipeline = StandardRAGPipeline(app_config, DEFAULT_DEEPSEEK_V3)
    questions = read_questions(app_config.data.questions_file)
    existing = load_existing_records(output_path)

    tasks = [
        ("Standard_RAG", lambda q: std_pipeline.generate(q, prompt_template=STANDARD_RAG_TEMPLATE)),
        ("w/o_SEM", lambda q: ipf_pipeline.generate(q, prompt_template=TRM_ONLY_TEMPLATE)),
        ("w/o_TRM", lambda q: std_pipeline.generate(q, prompt_template=SEM_ONLY_TEMPLATE)),
        ("Full_IPF_RAG", lambda q: ipf_pipeline.generate(q, prompt_template=IPF_RAG_TEMPLATE)),
    ]

    for idx, question in enumerate(questions, start=1):
        for task_name, task_func in tasks:
            if (idx, task_name) in existing:
                continue
            answer = task_func(question)
            append_json_record(output_path, {
                "question_idx": idx,
                "question": question,
                "answer": answer,
                "model": DEFAULT_DEEPSEEK_V3.model_name,
                "RAG": task_name,
            })
            existing.add((idx, task_name))
