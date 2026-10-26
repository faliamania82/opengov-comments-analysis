"""Ταξινόμηση σχολίων με Gemini.

label : υπέρ / κατά / πρόταση / άλλο  (η ταξινόμηση της εκφώνησης)
stance: υπέρ / κατά / ουδέτερο / μικτό (γενική στάση απέναντι στη ρύθμιση)
Αποθηκεύει την πρόοδο μετά από κάθε παρτίδα και συνεχίζει από εκεί που έμεινε.
"""
import json
import os
import time

import pandas as pd
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash-lite"
BATCH = 15              # σχόλια ανά κλήση
PER_CONSULTATION = None   # πιλοτικά: 50 ανά διαβούλευση. None = όλα
PAUSE = 4               # δευτερόλεπτα ανάμεσα στις κλήσεις
OUT = "data/classified_unique.csv"

LABELS = {"υπέρ", "κατά", "πρόταση", "άλλο"}
STANCES = {"υπέρ", "κατά", "ουδέτερο", "μικτό"}

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

PROMPT = """Είσαι αναλυτής δημόσιων διαβουλεύσεων. Σου δίνω σχόλια πολιτών και
φορέων για νομοσχέδιο. Πολλά σχόλια παραθέτουν το κείμενο του νομοσχεδίου.
Για κάθε σχόλιο δώσε δύο πεδία.

"label" (ΜΙΑ τιμή):
- "υπέρ": στηρίζει συνολικά τη ρύθμιση.
- "κατά": διαφωνεί με τη ρύθμιση ή ζητά απόσυρση / ριζική αλλαγή.
- "πρόταση": το κύριο περιεχόμενο είναι συγκεκριμένη πρόταση αλλαγής,
  προσθήκης ή νέας διατύπωσης σε άρθρο.
- "άλλο": ερώτηση, ουδέτερο ή τεχνικό σχόλιο, άσχετο κείμενο, ή δεν είναι σαφές.

"stance" (ΜΙΑ τιμή): η γενική στάση απέναντι στη ρύθμιση:
"υπέρ", "κατά", "ουδέτερο" ή "μικτό" (π.χ. συμφωνεί σε μέρος και διαφωνεί σε άλλο).

Απάντησε ΜΟΝΟ με JSON, λίστα της μορφής:
[{"id": 0, "label": "πρόταση", "stance": "μικτό"}, ...]

Σχόλια:
"""


def short_text(t):
    """Κόβει πολύ μεγάλα σχόλια: αρχή και τέλος, με μια ένδειξη στη μέση."""
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
            label = r.get("label") if r.get("label") in LABELS else "?"
            stance = r.get("stance") if r.get("stance") in STANCES else "?"
            rows.append({"consultation_id": row["consultation_id"],
                         "comment_row_id": row["comment_row_id"],
                         "label": label, "stance": stance})
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"  ολοκληρώθηκαν {min(start + BATCH, len(todo))}/{len(todo)}")
        time.sleep(PAUSE)

    out = pd.read_csv(OUT, dtype=str)
    print("\nΣύνοψη ανά διαβούλευση:")
    print(pd.crosstab(out["consultation_id"], out["label"]))
    print(pd.crosstab(out["consultation_id"], out["stance"]))


if __name__ == "__main__":
    main()