"""Ένωση αποτελεσμάτων: ομάδες, ετικέτες, στάση, συνόψεις -> data/final_comments.parquet."""
import pandas as pd

comments = pd.read_parquet("data/clustered_comments.parquet")
comments["consultation_id"] = comments["consultation_id"].astype(str)
comments["comment_row_id"] = comments["comment_row_id"].astype(str)

# Ετικέτες του Gemini (ένα ανά μοναδικό κείμενο) -> μεταφορά σε όλα τα ίδια σχόλια
cls = pd.read_csv("data/classified_unique.csv", dtype=str)
cls = cls.merge(comments[["comment_row_id", "norm"]], on="comment_row_id", how="left")
cls = cls.drop_duplicates(["consultation_id", "norm"])[["consultation_id", "norm", "label", "stance"]]
final = comments.merge(cls, on=["consultation_id", "norm"], how="left")
final[["label", "stance"]] = final[["label", "stance"]].fillna("σύντομο")

# Τίτλοι και συνόψεις των ομάδων
summ = pd.read_csv("data/cluster_summaries.csv", dtype={"consultation_id": str})
summ = summ.rename(columns={"title": "cluster_title", "summary": "cluster_summary"})
final = final.merge(
    summ[["consultation_id", "cluster", "cluster_title", "cluster_summary"]],
    on=["consultation_id", "cluster"], how="left",
)
final["cluster_title"] = final["cluster_title"].fillna("Πολύ σύντομα σχόλια")

final.to_parquet("data/final_comments.parquet", index=False)
print(f"Αποθηκεύτηκαν {len(final)} σχόλια στο data/final_comments.parquet\n")

print("Στάση ανά διαβούλευση (όλα τα σχόλια, μαζί με τα επαναλαμβανόμενα):")
print(pd.crosstab(final["consultation_id"], final["stance"], normalize="index").round(2))
print("\nΕτικέτα ανά διαβούλευση:")
print(pd.crosstab(final["consultation_id"], final["label"], normalize="index").round(2))