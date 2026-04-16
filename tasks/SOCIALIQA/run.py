"""SOCIALIQA 评测脚本（基于结构化输出，三选一 A/B/C）。"""
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加父目录到路径以导入 src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from SOCIALIQA.prompts import get_template, build_prompt
from SOCIALIQA.metrics import compute_metrics

# 关闭不必要日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _story_to_text(story: Dict[str, Any]) -> str:
    if not isinstance(story, dict):
        return ""
    full_story = story.get("full_story", "")
    return str(full_story).strip()


def build_mcq_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """将样本转换为 3 选 1 结构。要求 1 个正确答案 + 2 个错误答案。"""
    ans = row.get("Answer")
    if not isinstance(ans, dict):
        return None

    correct_list = ans.get("Correct_Answer", [])
    wrong_list = ans.get("Wrong_Answer", [])

    if not isinstance(correct_list, list) or not isinstance(wrong_list, list):
        return None
    if len(correct_list) != 1 or len(wrong_list) != 2:
        return None

    correct = str(correct_list[0]).strip()
    wrong = [str(w).strip() for w in wrong_list]

    letters = ["A", "B", "C"]
    choices = {
        letters[0]: correct,
        letters[1]: wrong[0],
        letters[2]: wrong[1],
    }

    return {
        "story": _story_to_text(row.get("Story", {})),
        "question": str(row.get("Question", "")).strip(),
        "choices": choices,
        "gold_letter": "A",
    }


def preprocess_mcq(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    skipped = 0
    for row in data:
        mcq = build_mcq_from_row(row)
        if mcq is None:
            skipped += 1
            continue
        out = dict(row)
        out["_mcq"] = mcq
        valid.append(out)

    if skipped:
        print(f"Warning: skipped {skipped} rows (expected 1 Correct_Answer + 2 Wrong_Answer).")
    if not valid:
        raise RuntimeError("没有可评测样本：SOCIALIQA 需要 1 Correct_Answer + 2 Wrong_Answer。")
    return valid


def shuffle_mcq_options(mcq: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """打乱选项顺序，并同步更新 gold_letter。"""
    rng = random.Random(seed)
    letters = sorted(mcq["choices"].keys())
    texts = [mcq["choices"][l] for l in letters]
    old_gold_idx = letters.index(mcq["gold_letter"])

    indices = list(range(len(letters)))
    rng.shuffle(indices)

    new_choices: Dict[str, str] = {}
    new_gold = mcq["gold_letter"]
    for new_pos, old_idx in enumerate(indices):
        new_choices[letters[new_pos]] = texts[old_idx]
        if old_idx == old_gold_idx:
            new_gold = letters[new_pos]

    return {**mcq, "choices": new_choices, "gold_letter": new_gold}


def main():
    # 加载数据集配置
    dataset_config = runner.load_dataset_config("tasks/SOCIALIQA/config.yaml")

    # 加载实验配置
    experiment_config = runner.load_experiment_config("experiment_config.yaml")

    schema = dataset_config["schema"]
    prompt_method = dataset_config["default_prompt"]
    template = get_template(prompt_method)

    # 创建 LLM 客户端
    client = runner.create_llm_client(experiment_config["llm_config"])

    # 加载数据
    data = runner.load_and_limit_data(
        subset=dataset_config["subset"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )

    print(f"Loaded {len(data)} raw rows from {dataset_config['subset']}")
    data = preprocess_mcq(data)

    repeats = experiment_config["repeats"]
    print(f"MCQ samples: {len(data)}")
    print(f"Prompt method: {prompt_method}")
    print(f"Repeats: {repeats} (each with different option shuffle)")

    all_prompts: List[str] = []
    repeat_data: List[List[Dict[str, Any]]] = []

    for i in range(repeats):
        shuffled_rows: List[Dict[str, Any]] = []
        for j, row in enumerate(data):
            shuffled_mcq = shuffle_mcq_options(row["_mcq"], seed=42 * (i + 1) + j)
            shuffled_row = dict(row)
            shuffled_row["_mcq"] = shuffled_mcq
            shuffled_rows.append(shuffled_row)
            all_prompts.append(build_prompt(template, shuffled_row))
        repeat_data.append(shuffled_rows)

    print(f"Running inference ({len(all_prompts)} prompts)...")
    results = client.batch_generate_structure(all_prompts, schema)

    n = len(data)
    all_predictions: List[List[str]] = []
    all_metrics: List[Dict[str, Any]] = []
    all_gold: List[List[str]] = []

    for i in range(repeats):
        start = i * n
        end = start + n
        repeat_results = results[start:end]
        rows = repeat_data[i]
        predictions = [getattr(r, "answer", "") for r in repeat_results]
        all_predictions.append(predictions)

        metrics = compute_metrics(predictions, rows)
        all_metrics.append(metrics)
        all_gold.append([row["_mcq"]["gold_letter"] for row in rows])

        print(
            f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, "
            f"Correct={metrics['correct']}/{metrics['total']}"
        )

    runner.save_common_results(
        dataset_name=dataset_config["dataset"],
        model=experiment_config["llm_config"]["model_name"],
        prompt_method=prompt_method,
        all_predictions=all_predictions,
        gold_answers=all_gold,
        all_metrics=all_metrics,
        results_path=experiment_config["results_path"],
        dataset_config=dataset_config,
        experiment_config=experiment_config,
    )

    runner.print_summary_stats(all_metrics, repeats, n)


if __name__ == "__main__":
    main()
