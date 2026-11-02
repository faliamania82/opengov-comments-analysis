"""Δείγμα αξιολόγησης: 42 σχόλια χωρίς τις ετικέτες του μοντέλου, σε Excel με λίστες επιλογής."""
import pandas as pd
from openpyxl.styles import Alignment
from openpyxl.worksheet.datavalidation import DataValidation

df = pd.read_parquet("data/final_comments.parquet")
df = df[df["stance"] != "σύντομο"].drop_duplicates(["consultation_id", "norm"])
sample = (df.groupby("consultation_id", group_keys=False).sample(14, random_state=7)
            .sample(frac=1, random_state=1).reset_index(drop=True))
sample.insert(0, "id", range(1, len(sample) + 1))

# Κλειδί με τις ετικέτες του μοντέλου: ΜΗΝ το ανοίξεις πριν τελειώσεις τον χαρακτηρισμό
sample[["id", "consultation_id", "stance", "has_proposal"]].rename(
    columns={"stance": "model_stance", "has_proposal": "model_proposal"}
).to_csv("data/eval_key.csv", index=False)

out = sample[["id", "content"]].copy()
out["content"] = out["content"].map(lambda t: " ".join(str(t).split())[:2500])
out["στάση"] = ""
out["πρόταση"] = ""

with pd.ExcelWriter("data/eval_sample.xlsx", engine="openpyxl") as w:
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
print("Δημιουργήθηκε το data/eval_sample.xlsx με", len(out), "σχόλια")