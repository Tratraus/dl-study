from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    roc_curve,
)


def validate_threshold_inputs(y_true, y_prob):
    """检查multi-hot真值与概率矩阵。"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_true.ndim != 2 or y_prob.ndim != 2:
        raise ValueError("y_true和y_prob必须是二维[N, K]")
    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"shape不一致: y_true={y_true.shape}, y_prob={y_prob.shape}"
        )
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true只能包含0和1")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob不能包含NaN或inf")
    if (y_prob < 0.0).any() or (y_prob > 1.0).any():
        raise ValueError("y_prob必须位于[0, 1]")

    return y_true.astype(np.int64), y_prob.astype(np.float64)


def make_threshold_grid(start=0.05, stop=0.95, step=0.05):
    """生成包含stop的阈值网格，并减少浮点误差。"""
    # TODO 1：使用np.arange从start生成到stop，需确保stop被包含。
    thresholds = np.arange(start, stop + step / 2, step)
    return np.round(thresholds, decimals=10)


def search_global_threshold(
    y_true,
    y_prob,
    thresholds=None,
    average="micro",
):
    """
    搜索所有标签共用的单一阈值。

    返回：
        best_threshold: float
        best_f1: float
        history: dict，含thresholds和f1_scores
    """
    y_true, y_prob = validate_threshold_inputs(y_true, y_prob)
    if thresholds is None:
        thresholds = make_threshold_grid()
    thresholds = np.asarray(thresholds, dtype=np.float64)

    if thresholds.ndim != 1 or len(thresholds) == 0:
        raise ValueError("thresholds必须是非空一维数组")
    if (thresholds < 0.0).any() or (thresholds > 1.0).any():
        raise ValueError("所有threshold必须位于[0, 1]")

    f1_scores = []
    for threshold in thresholds:
        # TODO 2：根据当前threshold得到0/1预测。
        y_pred = (y_prob >= threshold).astype(np.int64)

        # TODO 3：使用f1_score，average由参数传入，zero_division=0。
        score = f1_score(y_true, y_pred, average=average, zero_division=0)
        f1_scores.append(float(score))

    f1_scores = np.asarray(f1_scores)

    # np.argmax在并列时返回第一个位置，即选择较小的并列阈值。
    # TODO 4：找到f1_scores中最大值的索引。
    best_idx = np.argmax(f1_scores)

    return {
        "best_threshold": float(thresholds[best_idx]),
        "best_f1": float(f1_scores[best_idx]),
        "thresholds": thresholds,
        "f1_scores": f1_scores,
        "average": average,
    }


def plot_f1_threshold_curve(search_result, output_path):
    """绘制F1-阈值曲线。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    thresholds = search_result["thresholds"]
    f1_scores = search_result["f1_scores"]
    best_threshold = search_result["best_threshold"]
    best_f1 = search_result["best_f1"]

    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, f1_scores, marker="o", markersize=3)

    # TODO 5：用plt.scatter在(best_threshold, best_f1)处标出最优点。
    plt.scatter(best_threshold, best_f1, color="red", s=50, label="Best F1")

    plt.xlabel("Threshold")
    plt.ylabel(f"F1 ({search_result['average']})")
    plt.title("F1 vs. classification threshold")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def apply_per_label_thresholds(y_prob, thresholds):
    """
    对每个标签使用不同阈值。

    y_prob: shape [N, K]
    thresholds: shape [K]
    """
    y_prob = np.asarray(y_prob, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    if y_prob.ndim != 2:
        raise ValueError("y_prob必须是二维[N, K]")
    if thresholds.ndim != 1 or len(thresholds) != y_prob.shape[1]:
        raise ValueError("thresholds必须是长度为K的一维数组")

    # TODO 6：NumPy会把[K]广播到[N,K]，逐列比较对应阈值。
    return (y_prob >= thresholds).astype(np.int64)


def search_per_label_thresholds(
    y_true,
    y_prob,
    thresholds=None,
    fallback_threshold=0.5,
):
    """
    为每个标签独立选择使该标签F1最大的阈值。

    对验证集中全0或全1的标签，使用fallback_threshold。
    """
    y_true, y_prob = validate_threshold_inputs(y_true, y_prob)
    if thresholds is None:
        thresholds = make_threshold_grid()
    thresholds = np.asarray(thresholds, dtype=np.float64)

    if thresholds.ndim != 1 or len(thresholds) == 0:
        raise ValueError("thresholds必须是非空一维数组")
    if (thresholds < 0.0).any() or (thresholds > 1.0).any():
        raise ValueError("所有threshold必须位于[0, 1]")
    if not 0.0 <= fallback_threshold <= 1.0:
        raise ValueError("fallback_threshold必须位于[0, 1]")

    num_labels = y_true.shape[1]
    best_thresholds = np.full(num_labels, fallback_threshold, dtype=np.float64)
    best_f1_per_label = np.full(num_labels, np.nan, dtype=np.float64)
    valid_labels = np.zeros(num_labels, dtype=bool)

    for label_idx in range(num_labels):
        label_true = y_true[:, label_idx]
        label_prob = y_prob[:, label_idx]

        # TODO 7：如果真实标签只有一种值，保留fallback并跳过。
        if np.unique(label_true).size < 2:
            continue

        valid_labels[label_idx] = True
        label_scores = []

        for threshold in thresholds:
            # TODO 8：对当前一维标签概率做阈值化，再计算binary F1。
            label_pred = (label_prob >= threshold).astype(np.int64)
            score = f1_score(
                label_true,
                label_pred,
                average="binary",
                zero_division=0,
            )
            label_scores.append(float(score))

        # TODO 9：找到当前标签F1最大值的第一个索引。
        best_idx = np.argmax(label_scores)
        best_thresholds[label_idx] = thresholds[best_idx]
        best_f1_per_label[label_idx] = label_scores[best_idx]

    return {
        "best_thresholds": best_thresholds,
        "best_f1_per_label": best_f1_per_label,
        "valid_labels": valid_labels,
        "num_valid_labels": int(valid_labels.sum()),
        "fallback_threshold": float(fallback_threshold),
    }


def plot_label_roc_pr(y_true, y_prob, label_idx, output_path):
    """为一个同时包含正负样本的标签绘制ROC和PR曲线。"""
    y_true, y_prob = validate_threshold_inputs(y_true, y_prob)
    if not 0 <= label_idx < y_true.shape[1]:
        raise ValueError("label_idx越界")

    label_true = y_true[:, label_idx]
    label_prob = y_prob[:, label_idx]
    if np.unique(label_true).size < 2:
        raise ValueError("该标签必须同时包含正样本和负样本")

    # TODO 10：调用已导入的roc_curve和precision_recall_curve。
    fpr, tpr, _ = roc_curve(label_true, label_prob)
    precision, recall, _ = precision_recall_curve(label_true, label_prob)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(fpr, tpr, marker=".")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC curve - label {label_idx}")
    axes[0].grid(alpha=0.3)

    # TODO 11：PR曲线的x轴是recall，y轴是precision。
    axes[1].plot(recall, precision, marker=".")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR curve - label {label_idx}")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # 合成验证数据：6个样本、3个标签。
    y_true = np.array([
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 1],
    ])
    y_prob = np.array([
        [0.62, 0.18, 0.48],
        [0.58, 0.46, 0.32],
        [0.35, 0.61, 0.20],
        [0.21, 0.42, 0.67],
        [0.55, 0.30, 0.28],
        [0.25, 0.57, 0.52],
    ])

    result = search_global_threshold(
        y_true,
        y_prob,
        thresholds=make_threshold_grid(0.10, 0.90, 0.05),
        average="micro",
    )
    print("best_threshold:", result["best_threshold"])
    print("best_f1:", result["best_f1"])

    plot_f1_threshold_curve(
        result,
        "data/processed/day4_f1_threshold_curve.png",
    )

    # phase 2
    per_label_result = search_per_label_thresholds(
        y_true,
        y_prob,
        thresholds=make_threshold_grid(0.10, 0.90, 0.05),
        fallback_threshold=result["best_threshold"],
    )
    print("per-label thresholds:", per_label_result["best_thresholds"])
    print("per-label F1:", per_label_result["best_f1_per_label"])
    print("valid labels:", per_label_result["num_valid_labels"])

    plot_label_roc_pr(
        y_true,
        y_prob,
        label_idx=0,
        output_path="data/processed/day4_label0_roc_pr.png",
    )

    # phase 2：比较三种阈值策略
    fixed_05_pred = (y_prob >= 0.5).astype(np.int64)

    global_pred = (
        y_prob >= result["best_threshold"]
    ).astype(np.int64)

    per_label_pred = apply_per_label_thresholds(
        y_prob,
        per_label_result["best_thresholds"],
    )

    strategy_predictions = {
        "fixed_0.5": fixed_05_pred,
        "global_best": global_pred,
        "per_label_best": per_label_pred,
    }

    print("\nThreshold strategy comparison")

    for strategy_name, y_pred in strategy_predictions.items():
        micro_f1 = f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        )

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        print(
            f"{strategy_name:>15} | "
            f"micro_f1={micro_f1:.4f} | "
            f"macro_f1={macro_f1:.4f}"
        )