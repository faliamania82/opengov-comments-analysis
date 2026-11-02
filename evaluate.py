"""Αξιολόγηση: σύγκριση των δικών μου ετικετών με του μοντέλου στο δείγμα των 42 σχολίων."""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

key = pd.read_csv("data/eval_key.csv")
mine = pd.read_excel("data/eval_sample.xlsx")[["id", "στάση", "πρόταση"]]
m = key.merge(mine, on="id").dropna(subset=["στάση", "πρόταση"])
print(f"Χαρακτηρισμένα σχόλια: {len(m)}\n")

m["ok_stance"] = m["model_stance"] == m["στάση"]
m["ok_prop"] = m["model_proposal"] == m["πρόταση"]
print(f"Συμφωνία στάσης:   {m['ok_stance'].mean() * 100:.0f}%  "
      f"(Cohen's kappa {cohen_kappa_score(m['στάση'], m['model_stance']):.2f})")
print(f"Συμφωνία πρότασης: {m['ok_prop'].mean() * 100:.0f}%  "
      f"(Cohen's kappa {cohen_kappa_score(m['πρόταση'], m['model_proposal']):.2f})")

print("\nΑνά διαβούλευση (% συμφωνίας):")
print((m.groupby("consultation_id")[["ok_stance", "ok_prop"]].mean() * 100).round(0))

print("\nΠίνακας σύγχυσης στάσης (γραμμές: εγώ, στήλες: μοντέλο):")
print(pd.crosstab(m["στάση"], m["model_stance"]))

m.to_csv("data/evaluation_results.csv", index=False)
print("\nΑποθηκεύτηκε το data/evaluation_results.csv")