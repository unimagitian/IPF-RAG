from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


NORMAL_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "如果你是医生，请根据患者的病程描述给出医嘱建议。"),
    ("user", "患者病程描述：{input}"),
])

STANDARD_RAG_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一名呼吸科专家，需要为肺纤维化患者制定个性化医嘱。请结合IPF知识库上下文和药物知识库上下文，为当前患者提供诊疗方案。",
    ),
    (
        "user",
        "## IPF知识库上下文\n{ipf_context}\n\n## 药物知识库上下文\n{med_context}\n\n## 当前患者情况\n{input}",
    ),
])

IPF_RAG_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一名呼吸科专家，需要为肺纤维化患者制定个性化医嘱。请重点参考循证医学上下文中的相似案例，并为核心用药建议提供明确依据。",
    ),
    (
        "user",
        "## 循证医学上下文 (参考案例集)\n{reference_context}\n\n## 当前患者情况\n{input}\n\n请按以下结构输出：\n（一）具体诊断及依据\n（二）具体用药建议\n（三）非药物干预\n（四）随访计划",
    ),
])

TRM_ONLY_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一名呼吸科专家。请阅读按时间序列检索出的相似病例与医嘱记录，并结合当前患者病程直接给出诊疗方案。",
    ),
    (
        "user",
        "## 循证医学上下文\n{reference_context}\n\n## 当前患者情况\n{input}",
    ),
])

SEM_ONLY_TEMPLATE = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一名呼吸科专家，需要结合IPF知识库和药物知识库，为当前患者生成完整、结构化且有循证依据的诊疗方案。",
    ),
    (
        "user",
        "## IPF知识库上下文\n{ipf_context}\n\n## 药物知识库上下文\n{med_context}\n\n## 当前患者情况\n{input}",
    ),
])
