"""Δεύτερο δείγμα αξιολόγησης: 30 σχόλια (10 ανά διαβούλευση), χωρίς τα σχόλια του πρώτου δείγματος.
Το κλειδί περιέχει τις ετικέτες v2 και v3 του μοντέλου, ώστε να συγκριθούν στο ίδιο δείγμα."""
import pandas as pd
from openpyxl.styles import Alignment
from openpyxl.worksheet.datavalidation import DataValidation


def key(t):
    return " ".join(str(t).split())[:2500]


final = pd.read_parquet("data/final_comments.parquet")
final["comment_row_id"] = final["comment_row_id"].astype(str)
final["consultation_id"] = final["consultation_id"].astype(str)

v2 = pd.read_csv("data/classified_unique_v2.csv", dtype=str)
v2 = v2.merge(final[["comment_row_id", "norm"]], on="comment_row_id", how="left")
v2 = v2.drop_duplicates(["consultation_id", "norm"])[
    ["consultation_id", "norm", "stance", "has_proposal"]
].rename(columns={"stance": "v2_stance", "has_proposal": "v2_proposal"})

df = final[final["stance"] != "σύντομο"].drop_duplicates(["consultation_id", "norm"])
df = df.merge(v2, on=["consultation_id", "norm"], how="left")

old = set(pd.read_excel("data/eval_sample.xlsx")["content"].map(key))
df = df[~df["content"].map(key).isin(old)]

sample = (df.groupby("consultation_id", group_keys=False).sample(10, random_state=21)
            .sample(frac=1, random_state=2).reset_index(drop=True))
sample.insert(0, "id", range(1, len(sample) + 1))

# Κλειδί με τις ετικέτες του μοντέλου: ΜΗΝ το ανοίξεις πριν τελειώσεις τον χαρακτηρισμό
sample[["id", "consultation_id", "stance", "has_proposal", "v2_stance", "v2_proposal"]].rename(
    columns={"stance": "v3_stance", "has_proposal": "v3_proposal"}
).to_csv("data/eval2_key.csv", index=False)

out = sample[["id", "content"]].copy()
out["content"] = out["content"].map(key)
out["στάση"] = ""
out["πρόταση"] = ""

with pd.ExcelWriter("data/eval2_sample.xlsx", engine="openpyxl") as w:
    out.to_excel(w, index=False, sheet_name="δείγμα")
    ws = w.sheets["δείγμα"]
    ws.column_dimensions["A"].width = 6
    ws.column_dimensions["B"].width = 110
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 14
    for row in ws.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    n = len(out) + 1
    dv1 = DataValidation(type="list", formula1='"υπέρ,κατά,ουδέτερο"', allow_blank=True)
    dv2 = DataValidation(type="list", formula1='"ναι,όχι"', allow_blank=True)
    ws.add_data_validation(dv1)
    ws.add_data_validation(dv2)
    dv1.add(f"C2:C{n}")
    dv2.add(f"D2:D{n}")
print("Δημιουργήθηκε το data/eval2_sample.xlsx με", len(out), "σχόλια")