"""Σύνοψη της δεύτερης αξιολόγησης από το evaluation2_results.csv (χωρίς το Excel)."""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

m = pd.read_csv("data/evaluation2_results.csv")
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
print("\nΠίνακας σύγχυσης v3 (γραμμές: εγώ, στήλες: μοντέλο):")
print(pd.crosstab(m["στάση"], m["v3_stance"]))
print("\nΠίνακας σύγχυσης v2:")
print(pd.crosstab(m["στάση"], m["v2_stance"]))