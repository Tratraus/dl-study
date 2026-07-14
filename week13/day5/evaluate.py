import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import torch


def evaluate(model, loader, device, class_names=None):
    """
    统一的评估函数：跑一遍 loader，返回一个字典，包含常用指标。

    Returns:
        dict，包含：
            - accuracy: float
            - macro_f1: float
            - weighted_f1: float
            - per_class_f1: np.ndarray，shape (num_classes,)
            - per_class_recall: np.ndarray
            - confusion_matrix: np.ndarray
            - y_true, y_pred: np.ndarray（供后续画图/诊断用）
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for input_ids, mask, labels in loader:
            input_ids = input_ids.to(device)
            mask      = mask.to(device)
            logits = model(input_ids, mask)
            preds  = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    per_class_f1     = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    result = {
        'accuracy':          acc,
        'macro_f1':          macro_f1,
        'weighted_f1':       weighted_f1,
        'per_class_f1':      per_class_f1,
        'per_class_recall':  per_class_recall,
        'confusion_matrix':  cm,
        'y_true':            y_true,
        'y_pred':            y_pred,
    }

    if class_names is not None:
        result['report'] = classification_report(
            y_true, y_pred, target_names=class_names,
            zero_division=0, digits=3
        )

    return result


def verify_against_sklearn(result, y_true, y_pred):
    """
    自测函数：验证 evaluate() 内部计算的指标与直接调用 sklearn 是否一致。
    用于 Day 5 的正确性检查，正式训练流程中不需要调用。
    """
    ref_acc      = accuracy_score(y_true, y_pred)
    ref_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    assert np.isclose(result['accuracy'], ref_acc), \
        f"Accuracy 不一致：{result['accuracy']} vs {ref_acc}"
    assert np.isclose(result['macro_f1'], ref_macro_f1), \
        f"Macro F1 不一致：{result['macro_f1']} vs {ref_macro_f1}"

    print("✅ evaluate() 输出与 sklearn 参考值一致")
