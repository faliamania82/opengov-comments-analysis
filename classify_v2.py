"""Ταξινόμηση v2: στάση (υπέρ / κατά / ουδέτερο ) + αν περιέχει πρόταση (ναι / όχι).
Αποθηκεύει την πρόοδο μετά από κάθε παρτίδα και συνεχίζει από εκεί που έμεινε."""
import json
import os
import time

import pandas as pd
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash-lite"
BATCH = 15
PER_CONSULTATION = None   # None = όλα τα σχόλια
PAUSE = 4
ALLOW_MIXED = False        # False = μόνο υπέρ / κατά / ουδέτερο
OUT = "data/classified_unique_v2.csv"

STANCES = {"υπέρ", "κατά", "ουδέτερο"} | ({"μικτό"} if ALLOW_MIXED else set())

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

mixed_line = ""
if ALLOW_MIXED:
    mixed_line = ('- "μικτό": συμφωνεί σε σημαντικό μέρος και διαφωνεί σε άλλο, '
                  'χωρίς να υπερτερεί σαφώς η μία πλευρά.\n')
else:
    mixed_line = ('Αν υπάρχουν και θετικά και αρνητικά στοιχεία, διάλεξε την κυρίαρχη στάση.\n')

PROMPT = (
    "Είσαι αναλυτής δημόσιων διαβουλεύσεων. Σου δίνω σχόλια πολιτών και φορέων για "
    "νομοσχέδιο. Πολλά σχόλια παραθέτουν το κείμενο του νομοσχεδίου.\n"
    "Για κάθε σχόλιο δώσε δύο πεδία.\n\n"
    '"stance" (ΜΙΑ τιμή): η γενική στάση του σχολίου απέναντι στη ρύθμιση:\n'
    '- "υπέρ": στηρίζει τη ρύθμιση.\n'
    '- "κατά": διαφωνεί με τη ρύθμιση ή ζητά απόσυρση ή ριζική αλλαγή.\n'
    '- "ουδέτερο": ερώτηση, τεχνικό ή ουδέτερο σχόλιο, ή σχόλιο χωρίς σαφή στάση.\n'
    + mixed_line +
    '\n"has_proposal" (true ή false): true αν το σχόλιο περιέχει συγκεκριμένη πρόταση '
    "αλλαγής, προσθήκης ή νέας διατύπωσης σε διάταξη, ανεξάρτητα από τη στάση του. "
    "false αλλιώς.\n\n"
    "Απάντησε ΜΟΝΟ με JSON, λίστα της μορφής:\n"
    '[{"id": 0, "stance": "κατά", "has_proposal": true}, ...]\n\n'
    "Σχόλια:\n"
)


def short_text(t):
    t = " ".join(str(t).split())
    return t if len(t) <= 1000 else t[:700] + " [...] " + t[-300:]


def call(texts):
    items = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=PROMPT + items,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            return {d["id"]: d for d in json.loads(resp.text)}
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"  σφάλμα ({str(e)[:100]}). Ξαναδοκιμάζω σε {wait}s")
            time.sleep(wait)
    return {}


def yes_no(v):
    if v is True or str(v).strip().lower() in ("true", "ναι"):
        return "ναι"
    if v is False or str(v).strip().lower() in ("false", "όχι"):
        return "όχι"
    return "?"


def main():
    df = pd.read_parquet("data/clustered_comments.parquet")
    df = df[~df["short"]].drop_duplicates(["consultation_id", "norm"]).copy()
    df["comment_row_id"] = df["comment_row_id"].astype(str)
    if PER_CONSULTATION:
        df = df.sample(frac=1, random_state=1).groupby("consultation_id").head(PER_CONSULTATION)

    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT, dtype=str)["comment_row_id"])
    todo = df[~df["comment_row_id"].isin(done)]
    print(f"Προς ταξινόμηση: {len(todo)} σχόλια (έχουν ήδη γίνει {len(done)})")

    for start in range(0, len(todo), BATCH):
        chunk = todo.iloc[start:start + BATCH]
        result = call([short_text(t) for t in chunk["content"]])
        if not result:
            print("  Η παρτίδα απέτυχε, την παραλείπω (θα ξαναδοκιμαστεί στην επόμενη εκτέλεση)")
            continue
        rows = []
        for i, (_, row) in enumerate(chunk.iterrows()):
            r = result.get(i, {})
            stance = r.get("stance") if r.get("stance") in STANCES else "?"
            rows.append({"consultation_id": row["consultation_id"],
                         "comment_row_id": row["comment_row_id"],
                         "stance": stance,
                         "has_proposal": yes_no(r.get("has_proposal"))})
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"  ολοκληρώθηκαν {min(start + BATCH, len(todo))}/{len(todo)}")
        time.sleep(PAUSE)

    out = pd.read_csv(OUT, dtype=str)
    print("\nΣτάση ανά διαβούλευση:")
    print(pd.crosstab(out["consultation_id"], out["stance"]))
    print("\nΠεριέχει πρόταση:")
    print(pd.crosstab(out["consultation_id"], out["has_proposal"]))


if __name__ == "__main__":
    main()