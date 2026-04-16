"""SOCIALIQA prompts。"""
from typing import Any, Dict

PROMPTS = {
    "zero_shot": (
        "You are good at commonsense reasoning about social interactions.\n"
        "Read the story and question, then choose the best option.\n"
        "Return only one letter: A, B, or C."
    ),
}


def get_template(method: str) -> str:
    """获取指定方法的 prompt 模板。"""
    return PROMPTS.get(method, PROMPTS["zero_shot"])


def build_prompt(template: str, row: Dict[str, Any]) -> str:
    """构建 SOCIALIQA 的选择题 prompt。"""
    mcq = row["_mcq"]
    story = mcq["story"].strip()
    question = mcq["question"].strip()
    options = mcq["choices"]

    option_lines = [f"({letter}) {options[letter]}" for letter in sorted(options.keys())]
    option_block = "\n".join(option_lines)

    return (
        f"{template}\n\n"
        f"Story: {story}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_block}\n\n"
        f"Answer:"
    )
