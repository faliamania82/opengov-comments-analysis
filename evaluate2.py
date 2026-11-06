"""Αξιολόγηση σε νέο δείγμα: συμφωνία των ετικετών v2 και v3 με τους δικούς μου χαρακτηρισμούς."""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

key = pd.read_csv("data/eval2_key.csv")
mine = pd.read_excel("data/eval2_sample.xlsx")[["id", "στάση", "πρόταση"]]
m = key.merge(mine, on="id").dropna(subset=["στάση", "πρόταση"])
print(f"Χαρακτηρισμένα σχόλια: {len(m)}\n")

for name in ["v2", "v3"]:
    s, p = f"{name}_stance", f"{name}_proposal"
    ok_s = (m[s] == m["στάση"]).mean() * 100
    ok_p = (m[p] == m["πρόταση"]).mean() * 100
    k = cohen_kappa_score(m["στάση"], m[s])
    print(f"{name}: συμφωνία στάσης {ok_s:.0f}% (kappa {k:.2f}) · συμφωνία πρότασης {ok_p:.0f}%")

print("\nΠλήθος «κατά»: εγώ", (m["στάση"] == "κατά").sum(),
      "| v2", (m["v2_stance"] == "κατά").sum(),
      "| v3", (m["v3_stance"] == "κατά").sum())

print("\nΠίνακας σύγχυσης στάσης v3 (γραμμές: εγώ, στήλες: μοντέλο):")
print(pd.crosstab(m["στάση"], m["v3_stance"]))

m.to_csv("data/evaluation2_results.csv", index=False)
print("\nΑποθηκεύτηκε το data/evaluation2_results.csv")