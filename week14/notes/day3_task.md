# Week 14 Day 3：多标签评估指标

## 今日目标

实现一个可复用的多标签评估模块，能从模型输出的 logits 计算：

- Precision / Recall / F1（micro、macro、samples）
- 每个标签的 F1
- AUROC（micro、macro）
- AUPRC / Average Precision（micro、macro）

今天不训练真实模型，不搜索最优阈值。先用可控的合成数据把“评估尺子”校准。阈值搜索属于 Day 4。

> **任务方式说明**：本任务采用“带框架填空”方式，而不是要求从空白文件独立手写完整程序。下面已经给出导入、函数结构、返回字典和测试结构。你只需要依次完成标有 `TODO` 的位置。每个 TODO 都对应今天要理解的一个概念。

---

## 1. 从 logits 到二值预测

模型对每个样本输出50个 logits：

$$
Z \in \mathbb{R}^{N \times K}, \qquad K=50
$$

`BCEWithLogitsLoss` 训练时直接接收 logits，但评估时需要先转成每个标签独立的概率：

$$
P = \sigma(Z)
$$

再用阈值 $t$ 得到二值预测：

$$
\hat{Y}_{ij}=\mathbb{1}(P_{ij} \ge t)
$$

Day 3 固定使用 `threshold=0.5`。不要在模型最后一层加 sigmoid；sigmoid 只在评估或推理时调用。

```python
probs = torch.sigmoid(logits)
preds = (probs >= threshold).to(torch.int64)
```

---

## 2. Precision、Recall 和 F1

对一个标签：

$$
\text{Precision}=\frac{TP}{TP+FP}
$$

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

$$
F1=\frac{2\cdot \text{Precision}\cdot \text{Recall}}
{\text{Precision}+\text{Recall}}
$$

- Precision 回答：预测为正的标签中，有多少是真的？
- Recall 回答：真实正标签中，有多少被找到？
- F1 是 Precision 与 Recall 的调和平均，任一边很低时 F1 都会受到明显惩罚。

---

## 3. micro、macro 和 samples 在“平均什么”

### 3.1 Micro average：先汇总，再计算

Micro 把 $N\times K$ 个二分类结果全部放在一起，先求全局 TP/FP/FN，再计算指标。

它回答：**所有标签决策总体做得如何？**

高频标签产生的决策数更多，因此对 micro 影响更大。

### 3.2 Macro average：先按标签计算，再平均

先对50个标签分别计算 F1，再做算术平均：

$$
F1_{macro}=\frac{1}{K}\sum_{k=1}^{K}F1_k
$$

它回答：**如果每个GO标签都具有同等重要性，模型做得如何？**

低频标签与高频标签权重相同，所以 macro F1 对长尾失败更敏感。

### 3.3 Samples average：先按样本计算，再平均

先对每条蛋白质的标签集计算 F1，再对 $N$ 个样本平均。

它回答：**对一条典型蛋白质，其功能标签集预测得如何？**

`samples` 只适用于多标签任务，它和 macro/micro 的聚合轴不同。

---

## 4. 为什么不使用普通 accuracy 作为主指标

在50标签任务中，每个样本平均只有2.25个正标签，大部分位置都是0。一个把所有标签都预测为0的模型，按位置计算的 accuracy 仍可能很高，但它没有找到任何功能。

因此本周主要报告：

- `micro_f1`
- `macro_f1`
- `samples_f1`
- `per_label_f1`

Accuracy 可作补充，但不作主要模型选择依据。

---

## 5. AUROC 和 AUPRC

F1 依赖于某个确定阈值；AUROC 和 AUPRC 直接使用概率，衡量不同阈值下的整体排序能力。

- AUROC：衡量正样本排在负样本前面的能力。
- AUPRC：总结 Precision-Recall 曲线，对正样本稀少的长尾标签通常更有解释力。

注意：如果某个标签在当前评估集中只有一个类别（全0或全1），该标签的 AUROC 没有定义。评估函数不能因此整体崩溃，应忽略该标签后对其余有效标签求平均，并返回有效标签数。

---

## 6. 今日代码任务

新建：

```text
week14/tools/multilabel_metrics.py
week14/tests/test_day3_metrics.py
```

### 6.1 复制代码框架

先把下面的完整框架复制到 `week14/tools/multilabel_metrics.py`。不要删除已有结构，只填写 `TODO 1~8`。

```python
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
    probs_tensor = ______________________________

    # TODO 2：把概率与threshold比较，并转换为整数0/1。
    preds_tensor = ______________________________

    probs = probs_tensor.detach().cpu().numpy()
    preds = preds_tensor.detach().cpu().numpy()
    return probs, preds


def _validate_inputs(y_true, y_prob):
    """统一检查评估函数的输入，避免错误数据产生看似正常的指标。"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # TODO 3：检查二者都是二维数组，并检查shape完全一致。
    if __________________________________________:
        raise ValueError("y_true和y_prob必须都是二维数组[N, K]")
    if __________________________________________:
        raise ValueError(
            f"shape不一致: y_true={y_true.shape}, y_prob={y_prob.shape}"
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true只能包含0和1")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob不能包含NaN或inf")

    # TODO 4：检查所有概率都位于[0, 1]。
    if __________________________________________:
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
        if __________________________________________:
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
    y_pred = __________________________________________

    # 这两项不需要阈值，直接使用真实标签和预测概率。
    # TODO 7：填写micro AUROC和micro AUPRC对应的sklearn调用。
    auroc_micro = ______________________________________
    auprc_micro = ______________________________________

    auroc_macro, auprc_macro, valid_auroc_labels = _macro_auc_scores(
        y_true, y_prob
    )

    # TODO 8：仿照已经写好的micro行，补全macro和samples行。
    return {
        "precision_micro": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "precision_macro": ______________________________________,
        "precision_samples": ____________________________________,
        "recall_micro": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "recall_macro": _________________________________________,
        "recall_samples": _______________________________________,
        "f1_micro": f1_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "f1_macro": _____________________________________________,
        "f1_samples": ___________________________________________,
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
```

填空时遵循以下提示：

| TODO | 要填写的概念 | 可参考的位置 |
|---|---|---|
| 1 | sigmoid | 本文第1节 |
| 2、6 | `概率 >= 阈值` | 本文第1节 |
| 3 | `ndim` 与 `shape` 检查 | `_validate_inputs` 的报错信息 |
| 4 | 概率上下界检查 | 条件需捕获“小于0或大于1” |
| 5 | 一个数组中不同值的数量 | 可使用 `np.unique` |
| 7 | `roc_auc_score` 与 `average_precision_score` | 已导入的两个函数 |
| 8 | micro示范行 | 只改变函数名或 `average` 参数 |

如果某个 TODO 不知道如何填写，可以一次只询问一个 TODO；不需要自己从头重写整个函数。

### 6.2 复制测试框架

把下面框架复制到 `week14/tests/test_day3_metrics.py`。第一轮先完成 `TODO T1~T4`，跑通后再补扩展测试。

```python
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
    assert metrics["f1_micro"] == pytest.approx(__________)
    assert metrics["f1_macro"] == pytest.approx(__________)
    assert metrics["f1_samples"] == pytest.approx(__________)


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
    assert preds[0, 0] == __________
    assert preds[1, 1] == __________


def test_shape_mismatch_raises_value_error():
    y_true = np.zeros((2, 3))
    y_prob = np.zeros((2, 2))

    # TODO T3：填写应抛出的异常类型。
    with pytest.raises(__________):
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
    assert metrics["valid_auroc_labels"] == __________
    assert np.isfinite(metrics["auroc_macro"])
```

### 6.3 基础测试通过后的扩展任务

先不要同时手写所有测试。上面的4项通过后，再逐项增加：

1. **全零预测**：不产生 NaN，F1 按 `zero_division=0` 处理。
2. **micro 与 macro 不同**：构造高频标签预测好、低频标签预测差的样例，断言 `f1_micro > f1_macro`。
3. **samples 语义**：用小矩阵手算每个样本的 F1，与函数输出对齐。

扩展测试不要求你直接从空白开始写。我会在检查基础测试后，根据你的完成情况继续给出对应的小框架。

---

## 7. Sanity check

在模块的 `if __name__ == "__main__":` 中构造一个小型可解释样例，打印至少：

```text
f1_micro
f1_macro
f1_samples
auroc_macro
auprc_macro
valid_auroc_labels
```

你需要能用自然语言解释为什么这个样例的 micro、macro 和 samples 不同。

---

## 8. 运行方式

使用本仓库的 `dl-study` 虚拟环境：

```bash
cd week14
conda run -n dl-study python tools/multilabel_metrics.py
conda run -n dl-study python -m pytest tests/test_day3_metrics.py -v
```

完成 Day 3 前，还应一并运行全部测试：

```bash
conda run -n dl-study pytest -q
```

如果全量测试仍只因现有 `test_day2_loss.py` 的未定义变量失败，需单独记录；不要把它误判为 Day 3 指标模块失败。

---

## 9. 完成标准

- [ ] 能从 logits 正确得到 sigmoid 概率和二值预测。
- [ ] 能解释 micro、macro、samples 分别在哪个维度聚合。
- [ ] 评估函数返回 Precision/Recall/F1、per-label F1、AUROC 和 AUPRC。
- [ ] 对无正样本标签和零除问题有明确处理。
- [ ] Day 3 单元测试全部通过。
- [ ] 没有把阈值搜索提前混入 Day 3。

---

## 10. 今日输出问题

完成代码后，请回答：

1. 如果 `micro_f1` 明显高于 `macro_f1`，对当前 Top-50 长尾任务意味着什么？
2. 为什么“全部预测为0”可能拥有很高的按位 accuracy，却是一个无用模型？
3. Macro F1 中某个低频标签的 F1 为0，可能是模型能力问题，也可能是什么数据问题？
4. 为什么在当前高度不平衡任务中，AUPRC 通常比 AUROC 更值得关注？
5. 为什么 Day 3 要固定阈值0.5，而不是现在就选一个让 F1 最高的阈值？

---

## 下一步

Day 3 完成后进入 Day 4：在验证集上进行阈值搜索，对比全局阈值与每标签阈值，并绘制 F1-阈值曲线及 ROC/PR 曲线。

CLRS 图基础仍保留为 Day 7.5 独立任务，不占用 Day 3 的评估指标学习时间。
