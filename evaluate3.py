"""Αξιολόγηση της v4 σε νέο, ανεξάρτητο δείγμα 30 σχολίων (με υπερδειγματοληψία των «κατά»)."""
import pandas as pd

key = pd.read_csv("data/eval3_key.csv")
mine = pd.read_excel("data/eval3_sample.xlsx")[["id", "στάση", "πρόταση"]]
m = key.merge(mine, on="id").dropna(subset=["στάση", "πρόταση"])
print(f"Χαρακτηρισμένα σχόλια: {len(m)}\n")

k = m[m["v4_stance"] == "κατά"]
o = m[m["v4_stance"] != "κατά"]
print(f"Σχόλια που η v4 έβαλε «κατά»: {len(k)}")
print(f"  από αυτά, εγώ έβαλα «κατά» σε {(k['στάση'] == 'κατά').sum()} "
      f"(ακρίβεια {((k['στάση'] == 'κατά').mean() * 100) if len(k) else 0:.0f}%)")
print(f"Σχόλια που η v4 ΔΕΝ έβαλε «κατά»: {len(o)}")
print(f"  από αυτά, εγώ έβαλα «κατά» σε {(o['στάση'] == 'κατά').sum()}\n")

print(f"Συμφωνία πρότασης (όλα τα σχόλια): {(m['v4_proposal'] == m['πρόταση']).mean() * 100:.0f}%")
print("\nΠίνακας σύγχυσης στάσης (γραμμές: εγώ, στήλες: v4):")
print(pd.crosstab(m["στάση"], m["v4_stance"]))

m.to_csv("data/evaluation3_results.csv", index=False)
print("\nΑποθηκεύτηκε το data/evaluation3_results.csv")