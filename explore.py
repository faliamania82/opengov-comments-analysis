"""Εξερεύνηση του dataset glossAPI/OpenGov_Deliberations.

Κατεβάζει το αρχείο, "ισιώνει" τα φωλιασμένα σχόλια (ένα row ανά σχόλιο),
γράφει στατιστικά ανά διαβούλευση (stats.csv) και αποθηκεύει τα σχόλια
των επιλεγμένων διαβουλεύσεων σε μικρό αρχείο (data/selected_comments.parquet).
Το token διαβάζεται από τη μεταβλητή περιβάλλοντος HF_TOKEN.
"""
import os

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "glossAPI/OpenGov_Deliberations"
FILE = "OpenGov.parquet"
SELECTED_IDS = {"13", "8", "11"}  # διαβουλεύσεις που διαλέξαμε


def load_comments():
    """Επιστρέφει DataFrame με ένα row ανά σχόλιο."""
    path = hf_hub_download(
        repo_id=REPO,
        filename=FILE,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    pf = pq.ParquetFile(path)
    print("Schema:", pf.schema_arrow)

    rows = []
    # Διαβάζουμε σε μικρά τμήματα για να μην γεμίσει η μνήμη
    for batch in pf.iter_batches(batch_size=20):
        for consultation in batch.to_pylist():
            for article in consultation.get("articles") or []:
                for comment in article.get("comments") or []:
                    rows.append(
                        {
                            "consultation_id": str(consultation.get("consultation_id")),
                            "consultation_title": consultation.get("title"),
                            "article_id": article.get("article_id"),
                            "article_title": article.get("title"),
                            "comment_row_id": comment.get("comment_row_id"),
                            "date": comment.get("date"),
                            "content": comment.get("content"),
                        }
                    )
    return pd.DataFrame(rows)


def main():
    df = load_comments()
    print(f"Σύνολο σχολίων: {len(df)}")

    # Στατιστικά ανά διαβούλευση, με τα περισσότερα σχόλια πρώτα
    stats = (
        df.groupby(["consultation_id", "consultation_title"])
        .size()
        .reset_index(name="comments")
        .sort_values("comments", ascending=False)
    )
    stats.to_csv("stats.csv", index=False)
    print(stats.head(20).to_string(index=False))

    # Μικρό αρχείο μόνο για τις επιλεγμένες διαβουλεύσεις
    selected = df[df["consultation_id"].isin(SELECTED_IDS)]
    os.makedirs("data", exist_ok=True)
    selected.to_parquet("data/selected_comments.parquet", index=False)
    print(f"Αποθηκεύτηκαν {len(selected)} σχόλια στο data/selected_comments.parquet")


if __name__ == "__main__":
    main()