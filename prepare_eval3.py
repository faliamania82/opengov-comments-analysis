"""Τρίτο δείγμα αξιολόγησης (ανεξάρτητο): 30 σχόλια, 10 ανά διαβούλευση, που δεν ανήκουν
ούτε στο πρώτο (26) ούτε στο δεύτερο (30) δείγμα.
Τρέξε το ΜΟΝΟ αφού τελειώσει η v4 και ενημερωθεί το build_final.py,
ώστε το data/final_comments.parquet να έχει τις ετικέτες v4."""
import pandas as pd
from openpyxl.styles import Alignment
from openpyxl.worksheet.datavalidation import DataValidation


def key(t):
    return " ".join(str(t).split())[:2500]


final = pd.read_parquet("data/final_comments.parquet")
final["consultation_id"] = final["consultation_id"].astype(str)
base = final[final["stance"] != "σύντομο"].drop_duplicates(["consultation_id", "norm"]).copy()
base["k"] = base["content"].map(key)

# Δείγμα 1: από το αρχικό Excel (δεν πρέπει να έχει αλλοιωθεί)
old1 = set(pd.read_excel("data/eval_sample.xlsx")["content"].map(key))

# Δείγμα 2: ανακατασκευή με τον ίδιο κώδικα και σπόρους με το prepare_eval2.py
df2 = base[~base["k"].isin(old1)]
s2 = (df2.groupby("consultation_id", group_keys=False).sample(10, random_state=21)
         .sample(frac=1, random_state=2).reset_index(drop=True))
k2 = pd.read_csv("data/eval2_key.csv", dtype=str)
assert list(s2["consultation_id"]) == list(k2["consultation_id"]), \
    "Δεν μπόρεσα να ανακατασκευάσω το δείγμα 2. Σταμάτα και πες το στον βοηθό."
old2 = set(s2["k"])

# Δείγμα 3: νέα σχόλια
pool = base[~base["k"].isin(old1 | old2)]
s3 = (pool.groupby("consultation_id", group_keys=False).sample(10, random_state=33)
          .sample(frac=1, random_state=4).reset_index(drop=True))
s3.insert(0, "id", range(1, len(s3) + 1))

# Κλειδί με τις ετικέτες v4: ΜΗΝ το ανοίξεις πριν τελειώσεις τον χαρακτηρισμό
s3[["id", "consultation_id", "stance", "has_proposal"]].rename(
    columns={"stance": "v4_stance", "has_proposal": "v4_proposal"}
).to_csv("data/eval3_key.csv", index=False)

out = s3[["id", "content"]].copy()
out["content"] = out["content"].map(key)
out["στάση"] = ""
out["πρόταση"] = ""
instr = pd.DataFrame({"Οδηγίες χαρακτηρισμού": [
    "στάση = υπέρ: όλα όσα ΔΕΝ είναι ρητά κατά (στήριξη, προτάσεις βελτίωσης, εποικοδομητική κριτική).",
    "στάση = κατά: μόνο αν δηλώνει ρητά ότι δεν θέλει να προχωρήσει το νομοσχέδιο, ή θεωρεί ένα σημείο τόσο κρίσιμο ώστε να μην πρέπει να προχωρήσει χωρίς αυτό.",
    "στάση = ουδέτερο: μόνο ερωτήσεις, τεχνικά ή διαδικαστικά σχόλια, άσχετα ή ακατανόητα.",
    "πρόταση = ναι: το σχόλιο περιέχει συγκεκριμένη πρόταση αλλαγής, προσθήκης ή νέας διατύπωσης. Αλλιώς όχι.",
]})

with pd.ExcelWriter("data/eval3_sample.xlsx", engine="openpyxl") as w:
    out.to_excel(w, index=False, sheet_name="δείγμα")
    instr.to_excel(w, index=False, sheet_name="οδηγίες")
    ws = w.sheets["δείγμα"]
    w.sheets["οδηγίες"].column_dimensions["A"].width = 150
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
print("Δημιουργήθηκε το data/eval3_sample.xlsx με", len(out), "σχόλια")