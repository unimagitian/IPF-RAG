from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MedicalTextIndex:
    first_course: int = 0
    course_end: int = 0
    long_term_md: int = 0
    long_term_md_end: int = 0
    temporary_md: int = 0
    temporary_md_end: int = 0
    discharge_summary: int = 0
    case_id: int = 0
