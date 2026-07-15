"""
week14/tests/test_day1_data.py
------------------------------------------------------------
Day1数据构造正确性校验。shift-left原则：在训练前拦住数据层面错误。

用法:
    pytest tests/test_day1_data.py -v
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

PROCESSED_CSV = Path(__file__).parent.parent / "data/processed/multilabel_topk.csv"
LABEL_SPACE_TXT = Path(__file__).parent.parent / "data/processed/label_space.txt"


@pytest.fixture(scope="module")
def processed_df():
    if not PROCESSED_CSV.exists():
        pytest.skip(f"{PROCESSED_CSV} 不存在，请先运行 analyze_label_frequency.py")
    df = pd.read_csv(PROCESSED_CSV)
    df["go_list_filtered"] = df["go_list_filtered"].apply(ast.literal_eval)
    return df


@pytest.fixture(scope="module")
def label_space():
    if not LABEL_SPACE_TXT.exists():
        pytest.skip(f"{LABEL_SPACE_TXT} 不存在，请先运行 analyze_label_frequency.py")
    with open(LABEL_SPACE_TXT) as f:
        return [line.strip() for line in f if line.strip()]


def test_every_sample_has_at_least_one_label(processed_df):
    lengths = processed_df["go_list_filtered"].apply(len)
    assert (lengths >= 1).all(), "存在样本没有任何正标签"


def test_no_duplicate_entries(processed_df):
    assert processed_df["Entry"].is_unique, "存在重复的蛋白Entry"


def test_sequence_not_empty(processed_df):
    assert (processed_df["sequence"].str.len() > 0).all()


def test_labels_within_defined_space(processed_df, label_space):
    label_set = set(label_space)
    for go_list in processed_df["go_list_filtered"]:
        assert set(go_list).issubset(label_set), f"标签越界: {go_list}"


def test_multihot_shape_consistency(processed_df, label_space):
    label_index = {label: i for i, label in enumerate(label_space)}
    n_labels = len(label_space)
    sample_row = processed_df.iloc[0]
    vec = [0] * n_labels
    for go in sample_row["go_list_filtered"]:
        vec[label_index[go]] = 1
    assert len(vec) == n_labels
    assert sum(vec) == len(sample_row["go_list_filtered"])


def test_label_space_no_duplicates(label_space):
    assert len(label_space) == len(set(label_space))
