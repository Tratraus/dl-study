"""
week14/tools/fetch_uniprot_data.py
------------------------------------------------------------
从 UniProt REST API 拉取一批已审核（Swiss-Prot）人类蛋白，
包含序列 + GO Molecular Function 标注，构造多标签数据集。
数据来源：UniProt (https://www.uniprot.org)，公开免费，无需API key。

用法:
    python tools/fetch_uniprot_data.py --n 2000
"""

import argparse
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"


def fetch_uniprot_batch(query: str, fields: list, size: int = 500, max_retries: int = 3) -> pd.DataFrame:
    params = {
        "query": query,
        "fields": ",".join(fields),
        "format": "tsv",
        "size": size,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(UNIPROT_API, params=params, timeout=30)
            resp.raise_for_status()
            return pd.read_csv(StringIO(resp.text), sep="\t")
        except requests.exceptions.RequestException as e:
            print(f"[fetch_uniprot_data] 第{attempt+1}次请求失败: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("[fetch_uniprot_data] 多次重试后仍无法连接UniProt API")


def build_multilabel_dataset(n_target: int = 2000) -> pd.DataFrame:
    df = fetch_uniprot_batch(
        query="organism_id:9606 AND reviewed:true AND length:[50 TO 500]",
        fields=["accession", "sequence", "go_f"],
        size=n_target,
    )
    df = df.rename(columns={
        "Sequence": "sequence",
        "Gene Ontology (molecular function)": "go_terms",
    })
    df = df.dropna(subset=["sequence", "go_terms"]).reset_index(drop=True)

    def parse_go(s):
        terms = []
        for t in s.split(";"):
            t = t.strip()
            if "[" in t and t.endswith("]"):
                go_id = t.split("[")[-1].rstrip("]").strip()
                terms.append(go_id)
        return terms

    df["go_list"] = df["go_terms"].apply(parse_go)
    df = df[df["go_list"].apply(len) > 0].reset_index(drop=True)
    return df[["Entry", "sequence", "go_list"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--out", type=str, default="data/raw/uniprot_multilabel_raw.csv")
    args = parser.parse_args()

    df = build_multilabel_dataset(n_target=args.n)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"共拉取 {len(df)} 条蛋白记录")
    print(f"平均每条蛋白标签数: {df['go_list'].apply(len).mean():.2f}")
    print(f"已保存至: {out_path}")


if __name__ == "__main__":
    main()
