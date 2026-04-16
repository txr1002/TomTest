"""SOCIALIQA 数据集的输出 schema。"""
from typing import Literal

from pydantic import BaseModel


class MCQAnswer(BaseModel):
    """三选一答案 schema（选项字母）。"""
    answer: Literal["A", "B", "C"]


SCHEMAS = {
    "MCQAnswer": MCQAnswer,
}
