"""Συμφωνία v2 και v3 με τους δικούς μου χαρακτηρισμούς στα ίδια 26 σχόλια.
ΠΡΟΣΟΧΗ: το δείγμα χρησιμοποιήθηκε για να σχεδιαστούν οι ορισμοί της v3,
άρα το αποτέλεσμα της v3 δεν είναι ανεξάρτητη αξιολόγηση."""
import pandas as pd


def key(t):
    return " ".join(str(t).split())[:2500]


mine = pd.read_excel("data/eval_sample.xlsx")[["id", "content", "στάση", "πρόταση"]]
mine = mine.dropna(subset=["στάση", "πρόταση"])
mine["k"] = mine["content"].map(key)

v2 = pd.read_csv("data/eval_key.csv")[["id", "model_stance", "model_proposal"]]

final = pd.read_parquet("data/final_comments.parquet")
final["k"] = final["content"].map(key)
v3 = (final.drop_duplicates("k")[["k", "stance", "has_proposal"]]
           .rename(columns={"stance": "v3_stance", "has_proposal": "v3_proposal"}))

m = mine.merge(v2, on="id").merge(v3, on="k", how="left")
print(f"Σχόλια: {len(m)}  (βρέθηκαν στη v3: {m['v3_stance'].notna().sum()})\n")

for name, s, p in [("v2", "model_stance", "model_proposal"), ("v3", "v3_stance", "v3_proposal")]:
    print(f"{name}: συμφωνία στάσης {(m[s] == m['στάση']).mean() * 100:.0f}% · "
          f"συμφωνία πρότασης {(m[p] == m['πρόταση']).mean() * 100:.0f}%")

print("\nΜέτρηση «κατά» στο δείγμα: εγώ", (m["στάση"] == "κατά").sum(),
      "| v2", (m["model_stance"] == "κατά").sum(),
      "| v3", (m["v3_stance"] == "κατά").sum())
print("\nΠίνακας σύγχυσης v3 (γραμμές: εγώ, στήλες: v3):")
print(pd.crosstab(m["στάση"], m["v3_stance"]))