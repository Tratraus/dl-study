import numpy as np
import pytest
import torch

from tools.multilabel_metrics import (
    compute_multilabel_metrics,
    logits_to_predictions,
)


def test_perfect_prediction_has_f1_one():
    y_true = np.array([
        [1, 0, 1],
        [0, 1, 0],
    ])
    y_prob = np.array([
        [0.9, 0.1, 0.8],
        [0.2, 0.7, 0.1],
    ])

    metrics = compute_multilabel_metrics(y_true, y_prob, threshold=0.5)

    # TODO T1：分别断言micro、macro、samples F1都等于1.0。
    assert metrics["f1_micro"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["f1_samples"] == pytest.approx(1.0)


def test_logits_to_predictions():
    logits = torch.tensor([
        [0.0, 2.0],
        [-2.0, 0.0],
    ])

    probs, preds = logits_to_predictions(logits, threshold=0.5)

    assert probs.shape == (2, 2)
    assert preds.shape == (2, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # TODO T2：sigmoid(0)=0.5，而规则是>=0.5时预测为1。
    assert preds[0, 0] == 1
    assert preds[1, 1] == 1


def test_shape_mismatch_raises_value_error():
    y_true = np.zeros((2, 3))
    y_prob = np.zeros((2, 2))

    # TODO T3：填写应抛出的异常类型。
    with pytest.raises(ValueError):
        compute_multilabel_metrics(y_true, y_prob)


def test_label_without_both_classes_is_skipped_for_auc():
    # 第3列真实标签全为0，因此该列AUROC无定义，应被跳过。
    y_true = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])
    y_prob = np.array([
        [0.9, 0.2, 0.1],
        [0.8, 0.7, 0.2],
        [0.3, 0.8, 0.1],
        [0.1, 0.3, 0.2],
    ])

    metrics = compute_multilabel_metrics(y_true, y_prob)

    # TODO T4：3个标签中只有几个标签同时包含正样本和负样本？
    assert metrics["valid_auroc_labels"] == 2
    assert np.isfinite(metrics["auroc_macro"])