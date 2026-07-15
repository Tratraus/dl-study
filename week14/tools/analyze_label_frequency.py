"""
week14/tools/analyze_label_frequency.py
------------------------------------------------------------
读取 fetch_uniprot_data.py 拉取的原始数据，统计GO term频次，
辅助确定Top-K标签空间，并输出过滤后的multilabel数据集。

用法:
    python tools/analyze_label_frequency.py --k 50
"""

import argparse
import ast
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAW_CSV = Path("data/raw/uniprot_multilabel_raw.csv")
OUT_CSV = Path("data/processed/multilabel_topk.csv")
LABEL_SPACE_TXT = Path("data/processed/label_space.txt")
PLOT_PNG = Path("data/processed/label_frequency.png")


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["go_list"] = df["go_list"].apply(ast.literal_eval)
    return df


def analyze_and_filter(df: pd.DataFrame, k: int):
    all_labels = [label for labels in df["go_list"] for label in labels]
    counts = Counter(all_labels)

    print(f"标签总数(去重前): {len(all_labels)}")
    print(f"标签总数(去重后): {len(counts)}")
    print(f"Top-{k}标签频次:")
    for label, cnt in counts.most_common(k):
        print(f"  {label}: {cnt}")

    top_k_labels = set([label for label, _ in counts.most_common(k)])

    # 用Top-K标签过滤每个样本的标签列表
    df["go_list_filtered"] = df["go_list"].apply(
        lambda labels: [l for l in labels if l in top_k_labels]
    )
    before = len(df)
    df = df[df["go_list_filtered"].apply(len) > 0].reset_index(drop=True)
    after = len(df)
    print(f"\n过滤后剩余样本: {after}/{before} ({after/before*100:.1f}%)")
    print(f"平均每样本标签数(过滤后): {df['go_list_filtered'].apply(len).mean():.2f}")

    # 画频次分布图，观察衰减曲线，辅助判断K值是否合理
    labels_sorted, freqs_sorted = zip(*counts.most_common(100))
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(freqs_sorted) + 1), freqs_sorted, marker="o", markersize=3)
    plt.axvline(x=k, color="red", linestyle="--", label=f"K={k}")
    plt.xlabel("Label rank")
    plt.ylabel("Frequency")
    plt.title("GO label frequency distribution (Top 100)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PNG)
    print(f"频次分布图已保存至: {PLOT_PNG}")

    return df, top_k_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=50, help="Top-K标签空间大小")
    parser.add_argument("--raw", type=str, default=str(RAW_CSV))
    args = parser.parse_args()

    df = load_raw(Path(args.raw))
    df_filtered, top_k_labels = analyze_and_filter(df, args.k)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(OUT_CSV, index=False)

    with open(LABEL_SPACE_TXT, "w") as f:
        for label in sorted(top_k_labels):
            f.write(label + "\n")

    print(f"\n已保存过滤后数据集: {OUT_CSV}")
    print(f"已保存标签空间清单: {LABEL_SPACE_TXT}")


if __name__ == "__main__":
    main()
