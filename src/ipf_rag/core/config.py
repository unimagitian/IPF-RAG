from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class ModelConfig:
    model: str
    model_name: str
    api_base: str
    api_key_env: str
    streaming: bool = True
    temperature: float = 0.0

    def require_api_key(self) -> str:
        value = os.getenv(self.api_key_env)
        if not value:
            raise ValueError(f"Environment variable '{self.api_key_env}' is not set.")
        return value


@dataclass(slots=True)
class DataPaths:
    project_root: Path
    ipf_corpus_dir: Path = field(init=False)
    medication_corpus_dir: Path = field(init=False)
    medication_cache_json: Path = field(init=False)
    embedding_model_dir: Path = field(init=False)
    ipf_faiss_index_dir: Path = field(init=False)
    medication_faiss_index_dir: Path = field(init=False)
    questions_file: Path = field(init=False)
    outputs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.ipf_corpus_dir = self.project_root / "private_data" / "ipf_knowledge_base"
        self.medication_corpus_dir = self.project_root / "private_data" / "medication_information"
        self.medication_cache_json = self.medication_corpus_dir / "drug_dict_cache.json"
        self.embedding_model_dir = self.project_root / "models" / "bge-small-zh-v1.5"
        self.ipf_faiss_index_dir = self.project_root / "private_data" / "indexes" / "ipf_index"
        self.medication_faiss_index_dir = self.project_root / "private_data" / "indexes" / "medication_index"
        self.questions_file = self.project_root / "examples" / "test_questions.txt"
        self.outputs_dir = self.project_root / "outputs"


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    data: DataPaths = field(init=False)
    default_model: Optional[ModelConfig] = None
    retrieval_top_k: int = 7

    def __post_init__(self) -> None:
        self.data = DataPaths(project_root=self.project_root)

    @classmethod
    def from_project_root(cls, project_root: str | Path, default_model: Optional[ModelConfig] = None) -> "AppConfig":
        return cls(project_root=Path(project_root).resolve(), default_model=default_model)


DEFAULT_DEEPSEEK_V3 = ModelConfig(
    model="deepseek-chat",
    model_name="deepseek-V3",
    api_base="https://api.deepseek.com",
    api_key_env="IPF_RAG_API_KEY",
    streaming=True,
    temperature=0.0,
)
