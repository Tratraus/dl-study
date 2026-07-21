import numpy as np
import pytest

from tools.threshold_search import (
    make_threshold_grid,
    search_global_threshold,
    apply_per_label_thresholds,
    search_per_label_thresholds,
)


def test_threshold_grid_contains_both_ends():
    thresholds = make_threshold_grid(0.1, 0.9, 0.1)

    # TODO T1：检查第一个和最后一个阈值。
    assert thresholds[0] == pytest.approx(0.1)
    assert thresholds[-1] == pytest.approx(0.9)


def test_search_finds_expected_best_threshold():
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

    result = search_global_threshold(
        y_true, y_prob, thresholds=thresholds, average="micro"
    )

    # TODO T2：手动比较三个阈值后，填入最优阈值和F1。
    assert result["best_threshold"] == pytest.approx(0.5)
    assert result["best_f1"] == pytest.approx(1.0)


def test_tie_chooses_first_threshold():
    y_true = np.array([[1], [0]])
    y_prob = np.array([[0.9], [0.1]])
    thresholds = np.array([0.3, 0.5, 0.7])

    result = search_global_threshold(y_true, y_prob, thresholds=thresholds)

    # TODO T3：三个阈值都完美时，np.argmax会选哪一个？
    assert result["best_threshold"] == pytest.approx(0.3)


def test_shape_mismatch_raises_value_error():
    y_true = np.zeros((3, 2))
    y_prob = np.zeros((3, 3))

    # TODO T4：填写应抛出的异常类型。
    with pytest.raises(ValueError):
        search_global_threshold(y_true, y_prob)


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
        [1, 0],
        [1, 0],
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
    expected_thresholds = np.array([0.3, 0.5])
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
    assert result["best_thresholds"][1] == pytest.approx(0.5)
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
    with pytest.raises(ValueError):
        apply_per_label_thresholds(y_prob, np.array([0.5]))