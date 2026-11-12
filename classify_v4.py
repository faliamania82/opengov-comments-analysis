"""Ταξινόμηση v4: «κατά» μόνο ρητή άρνηση ή κρίσιμο σημείο· «ουδέτερο» μόνο ερωτήσεις,
τεχνικά και άσχετα· όλα τα υπόλοιπα «υπέρ» (όχι κατά, εποικοδομητικά).
Αποθηκεύει την πρόοδο μετά από κάθε παρτίδα και συνεχίζει από εκεί που έμεινε."""
import json
import os
import time

import pandas as pd
from google import genai
from google.genai import types

MODEL = "gemini-3.1-flash-lite"
BATCH = 15
PAUSE = 4
OUT = "data/classified_unique_v4.csv"
STANCES = {"υπέρ", "κατά", "ουδέτερο"}

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

PROMPT = (
    "Είσαι αναλυτής δημόσιων διαβουλεύσεων. Σου δίνω σχόλια πολιτών και φορέων για "
    "νομοσχέδιο. Πολλά σχόλια παραθέτουν το κείμενο του νομοσχεδίου.\n"
    "Για κάθε σχόλιο δώσε δύο πεδία.\n\n"
    '"stance" (ΜΙΑ τιμή): η στάση του σχολίου απέναντι στο νομοσχέδιο ως σύνολο.\n'
    '- "κατά": ΜΟΝΟ όταν το σχόλιο δηλώνει ρητά και ξεκάθαρα ότι δεν θέλει να προχωρήσει '
    "το νομοσχέδιο ή η ρύθμιση (π.χ. ζητά απόσυρση ή κατάργηση), Ή όταν θεωρεί το σημείο "
    "για το οποίο μιλά τόσο κρίσιμο ώστε το νομοσχέδιο να μην πρέπει να προχωρήσει χωρίς αυτό.\n"
    '- "ουδέτερο": ΜΟΝΟ όταν το σχόλιο είναι καθαρά ερώτηση, τεχνικό ή διαδικαστικό σχόλιο '
    "(π.χ. διόρθωση τυπογραφικού, αίτημα διευκρίνισης χωρίς άποψη), ή άσχετο με το νομοσχέδιο, "
    "ή ακατανόητο.\n"
    '- "υπέρ": ΟΛΑ τα υπόλοιπα, δηλαδή κάθε σχόλιο που δεν είναι ρητά "κατά" και δεν είναι '
    '"ουδέτερο". Εδώ ανήκουν σχόλια που στηρίζουν τη ρύθμιση, καθώς και σχόλια που προτείνουν '
    "βελτιώσεις, προσθήκες ή αλλαγές ή ασκούν εποικοδομητική κριτική χωρίς να ζητούν να μην "
    "προχωρήσει το νομοσχέδιο.\n"
    'ΣΗΜΑΝΤΙΚΟ: το ότι ένα σχόλιο ασκεί κριτική ή προτείνει αλλαγή ΔΕΝ το κάνει "κατά". '
    'Σε αμφιβολία ανάμεσα σε "κατά" και "υπέρ", διάλεξε "υπέρ". Σε αμφιβολία ανάμεσα σε '
    '"ουδέτερο" και "υπέρ", διάλεξε "υπέρ" εκτός αν το σχόλιο είναι σαφώς ερώτηση, τεχνικό ή άσχετο.\n\n'
    '"has_proposal" (true ή false): true αν το σχόλιο περιέχει συγκεκριμένη πρόταση '
    "αλλαγής, προσθήκης ή νέας διατύπωσης σε διάταξη, ανεξάρτητα από τη στάση του. "
    "false αλλιώς.\n\n"
    "Απάντησε ΜΟΝΟ με JSON, λίστα της μορφής:\n"
    '[{"id": 0, "stance": "υπέρ", "has_proposal": true}, ...]\n\n'
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