# Week 14 Day 4：多标签分类的阈值选择

## 今日目标

在验证数据上搜索分类阈值，理解阈值如何改变 Precision、Recall 和 F1，并实现：

- 全局阈值搜索：50个标签共用一个阈值。
- 按标签阈值搜索：每个标签单独选择阈值。
- F1-阈值曲线。
- 单个标签的 ROC 曲线和 PR 曲线。

今天仍然不训练真实模型。我们先用可控的合成验证数据让阈值搜索模块可运行、可测试；Day 6 获得真实验证集概率后，再复用同一模块获得真实阈值。

> **任务方式**：继续采用“代码框架 + TODO填空”。不需要从空白文件独立手写完整模块。第一轮只完成全局阈值搜索与基础测试；通过后再进入按标签阈值和绘图。

---

## 1. 阈值不是模型参数

模型输出概率 $p_{ij}$ 后，阈值 $t$ 将概率转换为最终决策：

$$
\hat{y}_{ij}=\mathbb{1}(p_{ij}\ge t)
$$

阈值不通过反向传播学习，而是在模型训练后，根据验证集表现选择的决策规则。

- 阈值降低：预测更多正标签，Recall 通常上升，但 FP 也可能增加，Precision 可能下降。
- 阈值升高：预测正标签更谨慎，Precision 通常上升，但会漏掉更多正例，Recall 可能下降。

F1 用来在 Precision 和 Recall 之间取平衡，因此常见的 F1-阈值曲线会在中间某处出现峰值。

---

## 2. 为什么只能在验证集上搜索

正确流程是：

```text
训练集：更新模型参数
验证集：选择阈值与其他超参数
测试集：冻结模型和阈值后，只做一次最终评估
```

如果直接在测试集上选使 F1 最高的阈值，等于利用测试答案调整决策规则，会造成数据泄漏。

---

## 3. 全局阈值与按标签阈值

### 3.1 全局阈值

所有标签共用一个 $t$。

优点：

- 简单，只选择一个数字。
- 过拟合风险相对较低。
- 容易解释和部署。

局限：高频和低频标签的概率分布可能不同，一个阈值未必适合所有标签。

### 3.2 按标签阈值

每个标签独立选择 $t_k$，最终得到长度为50的阈值向量。

优点：可以适应不同标签的概率分布，有机会提升 macro F1。

局限：需要选择50个数字。低频标签在验证集中可能只有几个正例，每标签阈值容易过拟合验证集噪声。

本周的实验顺序应为：

```text
基线：threshold=0.5
改进1：验证集最优全局阈值
改进2：验证集每标签阈值（需与全局阈值比较稳定性）
```

---

## 4. 第一阶段：全局阈值搜索

新建：

```text
week14/tools/threshold_search.py
week14/tests/test_day4_threshold.py
```

### 4.1 复制全局阈值代码框架

将下面内容复制到 `tools/threshold_search.py`，填写 `TODO 1~5`。

```python
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
    thresholds = __________________________________________
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
        y_pred = __________________________________________

        # TODO 3：使用f1_score，average由参数传入，zero_division=0。
        score = ___________________________________________
        f1_scores.append(float(score))

    f1_scores = np.asarray(f1_scores)

    # np.argmax在并列时返回第一个位置，即选择较小的并列阈值。
    # TODO 4：找到f1_scores中最大值的索引。
    best_idx = ____________________________________________

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
    __________________________________________

    plt.xlabel("Threshold")
    plt.ylabel(f"F1 ({search_result['average']})")
    plt.title("F1 vs. classification threshold")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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
```

### 4.2 TODO 提示

| TODO | 需要填写的概念 | 提示 |
|---|---|---|
| 1 | 阈值网格 | `np.arange(start, stop + step / 2, step)` |
| 2 | 概率阈值化 | 与 Day 3 的 `y_prob >= threshold` 一致 |
| 3 | F1计算 | 参照 Day 3 中 `f1_score` 的用法 |
| 4 | 最大值索引 | `np.argmax` |
| 5 | 标出最优点 | `plt.scatter(..., color="red", label=...)` |

---

## 5. 第一阶段测试框架

复制到 `tests/test_day4_threshold.py`，填写 `T1~T4`：

```python
import numpy as np
import pytest

from tools.threshold_search import (
    make_threshold_grid,
    search_global_threshold,
)


def test_threshold_grid_contains_both_ends():
    """
    测试目的：确认阈值网格同时包含start和stop。
    避免因np.arange的右边界规则或浮点误差遗漏最后一个候选阈值。
    """
    # Arrange + Act：生成一个起止点已知的阈值网格。
    thresholds = make_threshold_grid(0.1, 0.9, 0.1)

    # Assert：数组的首尾必须分别是0.1和0.9。
    # TODO T1：检查第一个和最后一个阈值。
    assert thresholds[0] == pytest.approx(__________)
    assert thresholds[-1] == pytest.approx(__________)


def test_search_finds_expected_best_threshold():
    """
    测试目的：确认搜索函数会遍历候选阈值并返回F1最高者。
    数据被刻意设计为阈值0.5时完美分类，因此正确答案可以人工确定。
    """
    # Arrange：构造0.5时所有位置都预测正确的小数据。
    y_true = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ])
    y_prob = np.array([
        [0.60, 0.20],
        [0.55, 0.30],
        [0.40, 0.70],
        [0.35, 0.65],
    ])
    thresholds = np.array([0.3, 0.5, 0.7])

    # Act：调用全局阈值搜索函数。
    result = search_global_threshold(
        y_true, y_prob, thresholds=thresholds, average="micro"
    )

    # Assert：与人工推导的已知正确答案比较。
    # TODO T2：手动比较三个阈值后，填入最优阈值和F1。
    assert result["best_threshold"] == pytest.approx(__________)
    assert result["best_f1"] == pytest.approx(__________)


def test_tie_chooses_first_threshold():
    """
    测试目的：确认多个阈值并列最优时的确定性规则。
    三个阈值都能完美分类，所以测试重点不是F1，而是np.argmax选择第一个最大值。
    """
    # Arrange：构造所有候选阈值都得到F1=1的场景。
    y_true = np.array([[1], [0]])
    y_prob = np.array([[0.9], [0.1]])
    thresholds = np.array([0.3, 0.5, 0.7])

    # Act：运行搜索。
    result = search_global_threshold(y_true, y_prob, thresholds=thresholds)

    # Assert：检查第一个并列最优阈值被选中。
    # TODO T3：三个阈值都完美时，np.argmax会选哪一个？
    assert result["best_threshold"] == pytest.approx(__________)


def test_shape_mismatch_raises_value_error():
    """
    测试目的：确认真实标签和预测概率shape不一致时立即报错。
    这个测试防止错位的样本或标签静默进入指标计算。
    """
    # Arrange：y_true有2个标签，y_prob却有3个标签。
    y_true = np.zeros((3, 2))
    y_prob = np.zeros((3, 3))

    # Act + Assert：调用函数时应抛出预期异常。
    # TODO T4：填写应抛出的异常类型。
    with pytest.raises(__________):
        search_global_threshold(y_true, y_prob)
```

### 第一阶段运行方式

你已经激活 `(dl-study)` 环境时：

```bash
cd ~/dl-study/week14
python tools/threshold_search.py
python -m pytest tests/test_day4_threshold.py -v
```

如果未激活环境：

```bash
conda run -n dl-study python tools/threshold_search.py
conda run -n dl-study python -m pytest tests/test_day4_threshold.py -v
```

---

## 6. 第二阶段：每标签阈值与 ROC/PR 曲线

第一阶段已通过。现在在 `tools/threshold_search.py` 中，放在 `plot_f1_threshold_curve()` 之后、`if __name__ == "__main__":` 之前，追加以下框架。

### 6.1 每标签阈值搜索

```python
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
    return ______________________________________________


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
        if ______________________________________________:
            continue

        valid_labels[label_idx] = True
        label_scores = []

        for threshold in thresholds:
            # TODO 8：对当前一维标签概率做阈值化，再计算binary F1。
            label_pred = __________________________________
            score = f1_score(
                label_true,
                label_pred,
                average="binary",
                zero_division=0,
            )
            label_scores.append(float(score))

        # TODO 9：找到当前标签F1最大值的第一个索引。
        best_idx = ________________________________________
        best_thresholds[label_idx] = thresholds[best_idx]
        best_f1_per_label[label_idx] = label_scores[best_idx]

    return {
        "best_thresholds": best_thresholds,
        "best_f1_per_label": best_f1_per_label,
        "valid_labels": valid_labels,
        "num_valid_labels": int(valid_labels.sum()),
        "fallback_threshold": float(fallback_threshold),
    }
```

### 6.2 单标签 ROC/PR 曲线

继续追加：

```python
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
    fpr, tpr, _ = _________________________________________
    precision, recall, _ = ________________________________

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
    axes[1].plot(__________________________________________)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR curve - label {label_idx}")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
```

### 6.3 TODO 提示

| TODO | 提示 |
|---|---|
| 6 | `(y_prob >= thresholds).astype(np.int64)` |
| 7 | `np.unique(label_true).size` |
| 8 | 与全局阈值的单次循环一致，但此时只有一列 |
| 9 | `np.argmax(label_scores)` |
| 10 | 两个函数都接收 `(label_true, label_prob)` |
| 11 | `recall, precision` |

---

## 6.4 第二阶段测试框架

在 `tests/test_day4_threshold.py` 末尾追加导入：

```python
from tools.threshold_search import (
    apply_per_label_thresholds,
    search_per_label_thresholds,
)
```

> 如果文件顶部已有一个 `from tools.threshold_search import (...)`，也可把这两个名字直接加进原导入括号，不必重复导入。

再追加测试：

```python
def test_apply_per_label_thresholds_uses_each_column_threshold():
    """
    测试目的：
    确认apply_per_label_thresholds()不是对整个矩阵使用同一阈值，
    而是把thresholds中的第k个值应用到y_prob的第k列。

    数据设计：
    - 第1列阈值较低(0.3)，0.4和0.6都应预测为1。
    - 第2列阈值较高(0.7)，0.4和0.6都应预测为0。

    这个测试保护的行为：NumPy广播必须按列使用不同阈值。
    """
    # Arrange：准备概率矩阵和两个明显不同的列阈值。
    y_prob = np.array([
        [0.4, 0.4],
        [0.6, 0.6],
    ])
    thresholds = np.array([0.3, 0.7])

    # Act：对两列分别使用各自阈值。
    preds = apply_per_label_thresholds(y_prob, thresholds)

    # Assert：手动写出每个位置比较后的已知答案。
    # TODO T5：第1列阈值0.3，第2列阈值0.7，手动写出预期矩阵。
    expected = np.array([
        [__, __],
        [__, __],
    ])
    np.testing.assert_array_equal(preds, expected)


def test_per_label_search_finds_different_thresholds():
    """
    测试目的：
    确认search_per_label_thresholds()会独立搜索每一列，
    并能为两个概率分布不同的标签选出不同阈值。

    数据设计：
    - 标签0的正例概率只有0.40和0.35，需要较低阈值才能全部找回。
    - 标签1有一个负例概率为0.40，阈值0.3会产生FP，
      因此需要更高阈值。

    这个测试保护的行为：不能把全局最优阈值简单复制K次。
    """
    # Arrange：构造两个最优分割位置不同的标签。
    y_true = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ])
    y_prob = np.array([
        [0.40, 0.40],
        [0.35, 0.20],
        [0.20, 0.80],
        [0.10, 0.70],
    ])
    thresholds = np.array([0.3, 0.5, 0.7])

    # Act：让被测函数逐标签搜索。
    result = search_per_label_thresholds(y_true, y_prob, thresholds=thresholds)

    # Assert：两个标签应各自找到手工可验证的最优阈值，且F1均为1。
    # TODO T6：逐列比较三个阈值，填入两个已知最优阈值。
    expected_thresholds = np.array([____, ____])
    np.testing.assert_allclose(result["best_thresholds"], expected_thresholds)
    np.testing.assert_allclose(result["best_f1_per_label"], [1.0, 1.0])


def test_invalid_label_uses_fallback_threshold():
    """
    测试目的：
    确认当某个标签在验证集中只有一种真值（全0或全1）时，
    函数不会为它伪造一个“最优阈值”，而是使用fallback_threshold。

    数据设计：
    - 第1列同时有0和1，是可搜索的有效标签。
    - 第2列全为0，没有正样本，无法可靠地选择F1阈值。

    这个测试保护的行为：无效标签必须回退，并在valid_labels中标记为False。
    """
    # Arrange：第1列有正负样本，第2列只有负样本。
    y_true = np.array([
        [1, 0],
        [0, 0],
        [1, 0],
        [0, 0],
    ])
    y_prob = np.array([
        [0.8, 0.1],
        [0.2, 0.2],
        [0.7, 0.3],
        [0.1, 0.4],
    ])

    # Act：明确指定回退阈值0.5。
    result = search_per_label_thresholds(
        y_true,
        y_prob,
        thresholds=np.array([0.3, 0.5, 0.7]),
        fallback_threshold=0.5,
    )

    # Assert：第2列保留fallback，且只有1个标签可以参与搜索。
    # TODO T7：第2列真值全0，应使用哪个回退阈值？
    assert result["best_thresholds"][1] == pytest.approx(____)
    assert result["valid_labels"].tolist() == [True, False]
    assert result["num_valid_labels"] == 1


def test_per_label_threshold_length_must_match_num_labels():
    """
    测试目的：
    确认函数会在计算前拦截阈值数量与标签数不一致的错误。

    数据设计：
    y_prob有2列，代表2个标签，但thresholds只提供1个值。

    这个测试保护的行为：
    不允许NumPy广播或shape错误静默产生看似正常的预测结果。
    """
    # Arrange：2个标签，但只提供1个阈值。
    y_prob = np.zeros((3, 2))

    # Act + Assert：调用函数时应立即抛出预期异常。
    # TODO T8：两个标签却只提供一个阈值，应抛出什么异常？
    with pytest.raises(__________):
        apply_per_label_thresholds(y_prob, np.array([0.5]))
```

### 6.5 第二阶段运行示例

在 `threshold_search.py` 的 `__main__` 末尾追加：

```python
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
```

运行：

```bash
python tools/threshold_search.py
python -m pytest tests/test_day4_threshold.py -v
```

预期新增图片：

```text
data/processed/day4_label0_roc_pr.png
```

---

## 7. 今日完成标准

### 第一阶段

- [x] 理解阈值降低与升高时 Precision/Recall 的变化方向。
- [x] 完成 `search_global_threshold()` 的5个 TODO。
- [x] 能解释为什么阈值只能在验证集上选择。
- [x] 4项基础测试全部通过。
- [x] 生成 `day4_f1_threshold_curve.png`。

### 第二阶段

- [ ] 实现每标签阈值搜索。
- [ ] 正确处理验证集中全0或全1标签。
- [ ] 生成一个标签的 ROC/PR 曲线。
- [ ] 对比三种阈值策略的指标。

---

## 8. 第一阶段输出问题

完成第一阶段后，请提供：

1. `best_threshold` 和 `best_f1` 的运行输出。
2. 4项 pytest 的结果。
3. 生成的 F1-阈值曲线。

并回答：

1. 阈值从0.5降到0.3时，TP、FP、FN通常会向什么方向变化？
2. 为什么在测试集上搜索最优阈值属于数据泄漏？
3. 如果0.35和0.40得到完全相同的最优 F1，当前代码会选哪一个？这个并列规则更偏向 Precision 还是 Recall？
4. 为什么每标签阈值可能提升 macro F1，却比全局阈值更容易过拟合？

---

## 下一步

先完成第一阶段，然后把代码、命令行输出、曲线和问答发来。我会检查后给出第二阶段的填空框架。
