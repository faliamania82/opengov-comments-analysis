"""Συμφωνία v2, v3 και v4 με τους δικούς μου χαρακτηρισμούς στο δείγμα των 30 σχολίων.
ΠΡΟΣΟΧΗ: οι χαρακτηρισμοί αυτοί χρησιμοποιήθηκαν για να οριστεί η v4, άρα το αποτέλεσμα
της v4 είναι αισιόδοξο και ΔΕΝ είναι ανεξάρτητη αξιολόγηση."""
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def key(t):
    return " ".join(str(t).split())[:2500]


# 1. Ανακατασκευή του δείγματος (ίδιος κώδικας και σπόροι με το prepare_eval2.py)
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

# 2. Έλεγχος ότι είναι το ίδιο δείγμα με αυτό που χαρακτήρισα
key2 = pd.read_csv("data/eval2_key.csv", dtype=str)
key2["id"] = key2["id"].astype(int)
assert list(sample["consultation_id"]) == list(key2["consultation_id"]), "διαφορετικό δείγμα (διαβούλευση)"
assert list(sample["v2_stance"]) == list(key2["v2_stance"]), "διαφορετικό δείγμα (v2)"
print("Το δείγμα ταυτίζεται με το eval2_key.csv\n")

# 3. Οι δικές μου ετικέτες και οι ετικέτες v3, v4
mine = pd.read_csv("data/evaluation2_results.csv")[["id", "στάση", "πρόταση"]]

v4 = pd.read_csv("data/classified_unique_v4.csv", dtype=str)
cl = pd.read_parquet("data/clustered_comments.parquet")[["comment_row_id", "norm"]]
cl["comment_row_id"] = cl["comment_row_id"].astype(str)
v4 = v4.merge(cl, on="comment_row_id", how="left")
v4 = v4.drop_duplicates(["consultation_id", "norm"])[
    ["consultation_id", "norm", "stance", "has_proposal"]
].rename(columns={"stance": "v4_stance", "has_proposal": "v4_proposal"})

m = (sample[["id", "consultation_id", "norm", "v2_stance", "v2_proposal"]]
     .merge(key2[["id", "v3_stance", "v3_proposal"]], on="id")
     .merge(mine, on="id")
     .merge(v4, on=["consultation_id", "norm"], how="left"))
print(f"Σχόλια: {len(m)}  (χωρίς ετικέτα v4: {m['v4_stance'].isna().sum()})\n")

for name in ["v2", "v3", "v4"]:
    s, p = f"{name}_stance", f"{name}_proposal"
    ok_s = (m[s] == m["στάση"]).mean() * 100
    ok_p = (m[p] == m["πρόταση"]).mean() * 100
    k = cohen_kappa_score(m["στάση"], m[s].fillna("?"))
    print(f"{name}: συμφωνία στάσης {ok_s:.0f}% (kappa {k:.2f}) · συμφωνία πρότασης {ok_p:.0f}%")

print("\nΠλήθος ανά στάση (εγώ | v2 | v3 | v4):")
for lab in ["υπέρ", "κατά", "ουδέτερο"]:
    print(f"  {lab}: {(m['στάση'] == lab).sum()} | {(m['v2_stance'] == lab).sum()} | "
          f"{(m['v3_stance'] == lab).sum()} | {(m['v4_stance'] == lab).sum()}")

print("\nΠίνακας σύγχυσης v4 (γραμμές: εγώ, στήλες: μοντέλο):")
print(pd.crosstab(m["στάση"], m["v4_stance"]))

m.to_csv("data/evaluation_v4_results.csv", index=False)
print("\nΑποθηκεύτηκε το data/evaluation_v4_results.csv")