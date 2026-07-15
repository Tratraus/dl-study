"""
week14/tools/fetch_uniprot_data.py（修正版）
------------------------------------------------------------
勘误说明：UniProt REST API 的 size 参数硬上限为 500，
超过会返回 400 Bad Request。本版本改为"size=500 + 游标分页"循环拉取。
"""

import argparse
import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"
MAX_PAGE_SIZE = 500  # UniProt官方硬上限，勿改大


def _parse_next_cursor_url(link_header: str) -> str | None:
    """
    从响应头 Link 中解析出下一页的完整URL。
    格式类似: <https://rest.uniprot.org/...&cursor=xxx>; rel="next"
    """
    if not link_header:
        return None
    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return match.group(1) if match else None


def fetch_uniprot_paginated(query: str, fields: list, n_target: int,
                             page_size: int = MAX_PAGE_SIZE,
                             max_retries: int = 3) -> pd.DataFrame:
    """
    使用官方推荐的游标分页方式循环拉取，直到达到n_target条或数据耗尽。
    """
    params = {
        "query": query,
        "fields": ",".join(fields),
        "format": "tsv",
        "size": min(page_size, MAX_PAGE_SIZE),
    }

    all_frames = []
    next_url = UNIPROT_API
    next_params = params
    fetched = 0

    while next_url and fetched < n_target:
        for attempt in range(max_retries):
            try:
                resp = requests.get(next_url, params=next_params, timeout=30)
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                print(f"[fetch_uniprot_data] 请求失败(第{attempt+1}次): {e}")
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError("[fetch_uniprot_data] 多次重试后仍无法连接UniProt API")

        df_page = pd.read_csv(StringIO(resp.text), sep="\t")
        all_frames.append(df_page)
        fetched += len(df_page)
        print(f"[fetch_uniprot_data] 已拉取 {fetched} 条...")

        # 解析下一页URL；官方分页机制通过Link header传递cursor
        next_url = _parse_next_cursor_url(resp.headers.get("Link", ""))
        next_params = None  # 下一页URL已包含所有查询参数，无需再传params

        if len(df_page) == 0:
            break  # 数据已耗尽

    df_all = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    return df_all.head(n_target)


def build_multilabel_dataset(n_target: int = 2000) -> pd.DataFrame:
    df = fetch_uniprot_paginated(
        query="organism_id:9606 AND reviewed:true AND length:[50 TO 500]",
        fields=["accession", "sequence", "go_f"],
        n_target=n_target,
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
    parser.add_argument("--n", type=int, default=2000, help="目标拉取条数")
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
