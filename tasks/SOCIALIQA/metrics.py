"""SOCIALIQA 数据集的 metrics 计算。"""
from typing import Any, Dict, List
import re


def _normalize_option(text: Any) -> str:
    if text is None:
        return ""
    s = str(text).strip().upper()
    if not s:
        return ""

    if s in {"A", "B", "C"}:
        return s

    # 兼容 "(A)"、"answer: B"、"option c" 等格式
    m = re.search(r"\b([ABC])\b", s)
    if m:
        return m.group(1)

    m = re.search(r"\(([ABC])\)", s)
    if m:
        return m.group(1)

    return ""


def compute_metrics(predictions: List[str], data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算 SOCIALIQA 的 metrics。

    - 整体准确率
    - 按 Meta.dimension[0] 分组准确率
    """
    gold_answers = [row["_mcq"]["gold_letter"] for row in data]

    correct = 0
    total = len(predictions)

    by_dimension: Dict[str, Dict[str, int]] = {}

    for pred, gold, row in zip(predictions, gold_answers, data):
        pred_norm = _normalize_option(pred)
        hit = bool(pred_norm) and pred_norm == gold
        if hit:
            correct += 1

        meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
        dimension = meta.get("dimension", "unknown")
        if isinstance(dimension, list) and dimension:
            dim_value = str(dimension[0])
        elif dimension:
            dim_value = str(dimension)
        else:
            dim_value = "unknown"

        if dim_value not in by_dimension:
            by_dimension[dim_value] = {"correct": 0, "total": 0}
        by_dimension[dim_value]["total"] += 1
        if hit:
            by_dimension[dim_value]["correct"] += 1

    accuracy = correct / total if total else 0

    secondary_metrics = {
        f"by_dimension.{dim}": (
            stats["correct"] / stats["total"] if stats["total"] else 0
        )
        for dim, stats in by_dimension.items()
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        **secondary_metrics,
        "by_dimension": {
            dim: (stats["correct"] / stats["total"] if stats["total"] else 0)
            for dim, stats in by_dimension.items()
        },
        "dimension_counts": {dim: stats["total"] for dim, stats in by_dimension.items()},
    }
