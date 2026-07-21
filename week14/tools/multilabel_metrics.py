import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def logits_to_predictions(logits: torch.Tensor, threshold: float = 0.5):
    """
    把模型输出的logits转换为概率和0/1预测。
    """
    if logits.ndim != 2:
        raise ValueError(f"logits必须是二维[N, K]，实际shape={tuple(logits.shape)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold必须位于[0, 1]")

    # TODO 1：使用torch.sigmoid把logits转换为概率。
    probs_tensor = torch.sigmoid(logits)

    # TODO 2：把概率与threshold比较，并转换为整数0/1。
    preds_tensor = (probs_tensor >= threshold).long()

    probs = probs_tensor.detach().cpu().numpy()
    preds = preds_tensor.detach().cpu().numpy()
    return probs, preds


def _validate_inputs(y_true, y_prob):
    """统一检查评估函数的输入，避免错误数据产生看似正常的指标。"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # TODO 3：检查二者都是二维数组，并检查shape完全一致。
    if y_true.ndim != 2 or y_prob.ndim != 2:
        raise ValueError("y_true和y_prob必须都是二维数组[N, K]")
    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"shape不一致: y_true={y_true.shape}, y_prob={y_prob.shape}"
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true只能包含0和1")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob不能包含NaN或inf")

    # TODO 4：检查所有概率都位于[0, 1]。
    if (y_prob < 0.0).any() or (y_prob > 1.0).any():
        raise ValueError("y_prob必须位于[0, 1]")

    return y_true.astype(np.int64), y_prob.astype(np.float64)


def _macro_auc_scores(y_true, y_prob):
    """
    逐标签计算AUROC和AUPRC。
    如果某一列全0或全1，该标签的AUROC没有定义，因此跳过。
    """
    auroc_values = []
    auprc_values = []

    for label_idx in range(y_true.shape[1]):
        label_true = y_true[:, label_idx]
        label_prob = y_prob[:, label_idx]

        # TODO 5：如果该标签的真实值只有一种取值，就continue跳过。
        if np.all(label_true == 0) or np.all(label_true == 1):
            continue

        auroc_values.append(roc_auc_score(label_true, label_prob))
        auprc_values.append(average_precision_score(label_true, label_prob))

    # 如果没有任何有效标签，就返回NaN；否则返回平均值。
    auroc_macro = float(np.mean(auroc_values)) if auroc_values else float("nan")
    auprc_macro = float(np.mean(auprc_values)) if auprc_values else float("nan")
    return auroc_macro, auprc_macro, len(auroc_values)


def compute_multilabel_metrics(y_true, y_prob, threshold: float = 0.5):
    """
    从真实multi-hot标签和预测概率计算多标签指标。
    """
    y_true, y_prob = _validate_inputs(y_true, y_prob)

    # TODO 6：根据固定阈值把概率转换成0/1预测。
    y_pred = (y_prob >= threshold).astype(np.int64)

    # 这两项不需要阈值，直接使用真实标签和预测概率。
    # TODO 7：填写micro AUROC和micro AUPRC对应的sklearn调用。
    auroc_micro = roc_auc_score(y_true, y_prob, average="micro")
    auprc_micro = average_precision_score(y_true, y_prob, average="micro")

    auroc_macro, auprc_macro, valid_auroc_labels = _macro_auc_scores(
        y_true, y_prob
    )

    # TODO 8：仿照已经写好的micro行，补全macro和samples行。
    return {
        "precision_micro": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "precision_samples": precision_score(
            y_true, y_pred, average="samples", zero_division=0
        ),
        "recall_micro": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_samples": recall_score(
            y_true, y_pred, average="samples", zero_division=0
        ),
        "f1_micro": f1_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_samples": f1_score(
            y_true, y_pred, average="samples", zero_division=0
        ),
        "f1_per_label": f1_score(
            y_true, y_pred, average=None, zero_division=0
        ),
        "auroc_micro": auroc_micro,
        "auroc_macro": auroc_macro,
        "auprc_micro": auprc_micro,
        "auprc_macro": auprc_macro,
        "valid_auroc_labels": valid_auroc_labels,
    }


if __name__ == "__main__":
    # 4个样本、3个标签的小型可解释示例。
    y_true = np.array([
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    y_prob = np.array([
        [0.90, 0.20, 0.40],
        [0.80, 0.45, 0.30],
        [0.30, 0.70, 0.20],
        [0.10, 0.60, 0.80],
    ])

    metrics = compute_multilabel_metrics(y_true, y_prob, threshold=0.5)
    for name in [
        "f1_micro",
        "f1_macro",
        "f1_samples",
        "auroc_macro",
        "auprc_macro",
        "valid_auroc_labels",
    ]:
        print(f"{name}: {metrics[name]}")