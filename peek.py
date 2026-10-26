import pandas as pd

a = pd.read_csv("data/classified_unique.csv", dtype=str)
b = pd.read_parquet("data/clustered_comments.parquet")
b["comment_row_id"] = b["comment_row_id"].astype(str)
m = a.merge(b[["comment_row_id", "content"]], on="comment_row_id")
for r in m.sample(12, random_state=3).itertuples():
    text = " ".join(str(r.content).split())[:350]
    print(f"[{r.label} | {r.stance}] {text}\n")