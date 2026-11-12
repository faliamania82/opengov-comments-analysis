"""Αξιολόγηση της v4 σε νέο, ανεξάρτητο δείγμα 30 σχολίων."""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

key = pd.read_csv("data/eval3_key.csv")
mine = pd.read_excel("data/eval3_sample.xlsx")[["id", "στάση", "πρόταση"]]
m = key.merge(mine, on="id").dropna(subset=["στάση", "πρόταση"])
print(f"Χαρακτηρισμένα σχόλια: {len(m)}\n")

ok_s = (m["v4_stance"] == m["στάση"]).mean() * 100
ok_p = (m["v4_proposal"] == m["πρόταση"]).mean() * 100
try:
    k = f"{cohen_kappa_score(m['στάση'], m['v4_stance']):.2f}"
except Exception:
    k = "—"
print(f"v4: συμφωνία στάσης {ok_s:.0f}% (kappa {k}) · συμφωνία πρότασης {ok_p:.0f}%")

print("\nΠλήθος ανά στάση (εγώ | v4):")
for lab in ["υπέρ", "κατά", "ουδέτερο"]:
    print(f"  {lab}: {(m['στάση'] == lab).sum()} | {(m['v4_stance'] == lab).sum()}")

print("\nΠίνακας σύγχυσης στάσης (γραμμές: εγώ, στήλες: v4):")
print(pd.crosstab(m["στάση"], m["v4_stance"]))

m.to_csv("data/evaluation3_results.csv", index=False)
print("\nΑποθηκεύτηκε το data/evaluation3_results.csv")