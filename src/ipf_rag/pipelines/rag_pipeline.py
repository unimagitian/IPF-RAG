from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from flashtext import KeywordProcessor
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from ipf_rag.core.config import AppConfig, ModelConfig
from ipf_rag.core.drug_dict import load_or_build_drug_dict
from ipf_rag.core.models import MedicalTextIndex
from ipf_rag.core.prompts import IPF_RAG_TEMPLATE, NORMAL_TEMPLATE, STANDARD_RAG_TEMPLATE

TIME_PATTERN = r"""
(
  (0[1-9]|1[0-2])\.
  (0[1-9]|[12][0-9]|3[01])
  \s
  ([01]\d|2[0-3]):
  ([0-5]\d)
)
|
(
  \d{4}年
  (0[1-9]|1[0-2])月
  (0[1-9]|[12][0-9]|3[01])日
  ([01]\d|2[0-3])时
  ([0-5]\d)分
)
|
(
  \d{4}年
  (0[1-9]|1[0-2])月
  (0[1-9]|[12][0-9]|3[01])日
  \s
  ([01]\d|2[0-3])时
  ([0-5]\d)分
)
"""


@dataclass(slots=True)
class RetrievedCase:
    symptom: Document | None
    long_term_order: Document | None
    temporary_order: Document | None
    discharge_summary: Document | None
    medication_information: Document | None


class BasePipeline:
    def __init__(self, app_config: AppConfig, model_config: ModelConfig):
        self.app_config = app_config
        self.model_config = model_config

    def _build_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model_config.model,
            openai_api_base=self.model_config.api_base,
            openai_api_key=self.model_config.require_api_key(),
            streaming=self.model_config.streaming,
            temperature=self.model_config.temperature,
        )

    def _build_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=str(self.app_config.data.embedding_model_dir),
            model_kwargs={"device": "cpu"},
        )

    def _load_retriever(self, index_path, top_k: int):
        vectordb = FAISS.load_local(
            folder_path=str(index_path),
            embeddings=self._build_embeddings(),
            allow_dangerous_deserialization=True,
        )
        retriever = vectordb.as_retriever()
        retriever.search_kwargs["k"] = top_k
        return retriever

    @staticmethod
    def _load_texts(directory) -> list[Document]:
        docs: list[Document] = []
        for file_path in sorted(directory.iterdir()):
            if file_path.is_file():
                docs.extend(TextLoader(str(file_path), encoding="utf-8").load())
        return docs


class StandardRAGPipeline(BasePipeline):
    def generate(self, message: str, top_k: int = 3, prompt_template=STANDARD_RAG_TEMPLATE) -> str:
        llm = self._build_llm()
        ipf_retriever = self._load_retriever(self.app_config.data.ipf_faiss_index_dir, top_k)
        med_retriever = self._load_retriever(self.app_config.data.medication_faiss_index_dir, top_k)
        ipf_context = "\n\n".join(doc.page_content for doc in ipf_retriever.invoke(message))
        med_context = "\n\n".join(doc.page_content for doc in med_retriever.invoke(message))
        chain = prompt_template | llm
        response = chain.invoke({"input": message, "ipf_context": ipf_context, "med_context": med_context})
        return response.content


class IPFRAGPipeline(BasePipeline):
    def __init__(self, app_config: AppConfig, model_config: ModelConfig):
        super().__init__(app_config, model_config)
        self._drug_dict = load_or_build_drug_dict(
            self.app_config.data.medication_corpus_dir / "Specific_information.txt",
            self.app_config.data.medication_cache_json,
        )
        self._drug_processor = KeywordProcessor()
        self._drug_processor.add_keywords_from_list(list(self._drug_dict.keys()))
        self._drug_processor.set_non_word_boundaries(set())

    def generate(self, message: str, top_k: int = 1, prompt_template=IPF_RAG_TEMPLATE) -> str:
        llm = self._build_llm()
        examples = self._mix_retrieval(message)[:top_k]
        if not examples:
            return "检索未找到相关病例，无法依据现有库提供参考。"
        reference_context = "\n".join(self._format_case_block(i, example) for i, example in enumerate(examples, start=1))
        chain = prompt_template | llm
        response = chain.invoke({"input": message, "reference_context": reference_context})
        return response.content

    def generate_without_rag(self, message: str, prompt_template=NORMAL_TEMPLATE) -> str:
        llm = self._build_llm()
        chain = prompt_template | llm
        response = chain.invoke({"input": message})
        return response.content

    def _mix_retrieval(self, message: str) -> list[RetrievedCase]:
        docs = self._load_texts(self.app_config.data.ipf_corpus_dir)
        indexes: list[MedicalTextIndex] = []
        splits = self._split_by_time_sequence(docs, indexes)
        retriever = self._load_retriever(self.app_config.data.ipf_faiss_index_dir, self.app_config.retrieval_top_k)
        matched_docs = retriever.invoke(message)
        cases: list[RetrievedCase] = []
        for doc in matched_docs:
            case_index = indexes[doc.metadata["id"]]
            original_timestamp = doc.metadata["timestamp"] or splits[case_index.first_course].metadata["timestamp"]
            long_idx = self._retrieve_by_time_sequence(case_index.long_term_md, case_index.long_term_md_end, original_timestamp, splits)
            temp_idx = self._retrieve_by_time_sequence(case_index.temporary_md, case_index.temporary_md_end, original_timestamp, splits)
            long_doc = splits[long_idx] if long_idx != -1 else None
            temp_doc = splits[temp_idx] if temp_idx != -1 else None
            if long_doc and len(long_doc.page_content) < 50:
                long_doc = self._retrieve_by_time_domain(long_idx, case_index, splits, True)
            if temp_doc and len(temp_doc.page_content) < 50:
                temp_doc = self._retrieve_by_time_domain(temp_idx, case_index, splits, False)
            discharge_doc = splits[case_index.discharge_summary]
            discharge_text = discharge_doc.page_content.split("出院诊断")[-1]
            discharge_summary = Document(page_content=f"出院诊断{discharge_text}")
            med_info = self._build_medication_document(long_doc, temp_doc)
            cases.append(RetrievedCase(doc, long_doc, temp_doc, discharge_summary, med_info))
        return cases

    def _build_medication_document(self, long_doc: Document | None, temp_doc: Document | None) -> Document:
        combined = f"{long_doc.page_content if long_doc else ''}\n{temp_doc.page_content if temp_doc else ''}"
        extracted = list(set(self._drug_processor.extract_keywords(combined)))
        if not extracted:
            return Document(page_content="未在医嘱中精确匹配到已知药物。")
        content = "\n\n".join(f"【{name}】\n{self._drug_dict[name]}" for name in extracted)
        return Document(page_content=content)

    def _format_case_block(self, rank: int, example: RetrievedCase) -> str:
        return (
            f">>> [参考案例 {rank}]\n"
            f"【相似特征】: {example.symptom.page_content if example.symptom else '无记录'}\n"
            f"【长期医嘱】: {example.long_term_order.page_content if example.long_term_order else '无记录'}\n"
            f"【临时医嘱】: {example.temporary_order.page_content if example.temporary_order else '无记录'}\n"
            f"【出院诊断】: {example.discharge_summary.page_content if example.discharge_summary else '无记录'}\n"
            f"【相关药品信息】:\n{example.medication_information.page_content if example.medication_information else '无记录'}\n"
            f"----------------------------------------"
        )

    @staticmethod
    def _safe_parse_time(time_str: str | None, year: int | None = None):
        if time_str is None:
            return None
        for fmt, has_year in [("%Y年%m月%d日%H时%M分", True), ("%m.%d %H:%M", False), ("%Y年%m月%d日 %H时%M分", True)]:
            try:
                dt = datetime.strptime(time_str, fmt)
                if not has_year:
                    dt = dt.replace(year=year or datetime.now().year)
                dt.strftime(fmt)
                return dt
            except ValueError:
                continue
        raise ValueError(f"无法解析时间字符串: {time_str}")

    @classmethod
    def _analyze_content(cls, content: str, year: int | None = None) -> dict:
        time_match = re.search(TIME_PATTERN, content, re.VERBOSE)
        time_str = time_match[0] if time_match else None
        dt = cls._safe_parse_time(time_str, year)
        return {"time": dt, "timestamp": dt.timestamp() if dt else None}

    @staticmethod
    def _update_index(content: str, counter: int, index: MedicalTextIndex) -> None:
        if re.search("长期医嘱", content) and index.long_term_md == 0:
            index.long_term_md = counter
            index.course_end = counter - 1
        if re.search("临时医嘱", content) and index.temporary_md == 0:
            index.temporary_md = counter
            index.long_term_md_end = counter - 1
        if re.search(r"出院小结", content) and index.discharge_summary == 0:
            index.discharge_summary = counter
            index.temporary_md_end = counter - 1

    @classmethod
    def _split_by_time_sequence(cls, docs: list[Document], indexes: list[MedicalTextIndex] | None = None) -> list[Document]:
        splits: list[Document] = []
        counter = 0
        for case_id, doc in enumerate(docs):
            chunks = doc.page_content.split("\n\n")
            first_time = cls._analyze_content(chunks[0]).get("time")
            index = MedicalTextIndex(first_course=counter, case_id=case_id)
            for chunk in chunks:
                metadata = cls._analyze_content(chunk, first_time.year if first_time else None)
                metadata["source"] = doc.metadata.get("source")
                metadata["id"] = case_id
                splits.append(Document(page_content=chunk, metadata=metadata))
                cls._update_index(chunk, counter, index)
                counter += 1
            if indexes is not None:
                indexes.append(index)
        return splits

    @staticmethod
    def _retrieve_by_time_sequence(left: int, right: int, original_timestamp, splits: list[Document]) -> int:
        left_bound, right_bound = left, right
        result = -1
        while left <= right:
            middle = (left + right) // 2
            target_timestamp = splits[middle].metadata["timestamp"]
            while target_timestamp is None and middle != right:
                middle += 1
                target_timestamp = splits[middle].metadata["timestamp"]
            if target_timestamp is None:
                break
            if target_timestamp > original_timestamp:
                result = middle
                right = middle - 1
            else:
                left = middle + 1
        if result == -1 and splits[right_bound].metadata["timestamp"] and abs(splits[right_bound].metadata["timestamp"] - original_timestamp) / 86400 < 1:
            result = right_bound
        if result == -1:
            return result
        delta = abs(splits[result].metadata["timestamp"] - original_timestamp) if splits[result].metadata["timestamp"] else 0
        while delta / 86400 > 2 and left_bound <= result <= right_bound:
            result -= 1
            if result < left_bound:
                return left_bound
            if not splits[result].metadata["timestamp"]:
                break
            delta = abs(splits[result].metadata["timestamp"] - original_timestamp)
        return result

    @staticmethod
    def _retrieve_by_time_domain(original_idx: int, index: MedicalTextIndex, splits: list[Document], is_long: bool) -> Document:
        upper_bound = index.long_term_md_end if is_long else index.temporary_md_end
        lower_bound = index.long_term_md if is_long else index.temporary_md
        current_day = splits[original_idx].metadata["time"].day
        page_content = splits[original_idx].page_content + "\n"
        count = 1
        upper_open = True
        lower_open = True
        step = 1
        while count < 3 and (upper_open or lower_open):
            if upper_open and original_idx + step <= upper_bound and splits[original_idx + step].metadata["time"] and splits[original_idx + step].metadata["time"].day == current_day:
                page_content += splits[original_idx + step].page_content + "\n"
                count += 1
            if count == 3:
                break
            if lower_open and original_idx - step >= lower_bound and splits[original_idx - step].metadata["time"] and splits[original_idx - step].metadata["time"].day == current_day:
                page_content = splits[original_idx - step].page_content + "\n" + page_content
                count += 1
            upper_open = bool(original_idx + step <= upper_bound and splits[original_idx + step].metadata["time"] and splits[original_idx + step].metadata["time"].day == current_day)
            lower_open = bool(original_idx - step >= lower_bound and splits[original_idx - step].metadata["time"] and splits[original_idx - step].metadata["time"].day == current_day)
            step += 1
        return Document(page_content=page_content)
