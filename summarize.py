"""Σύνοψη κάθε ομάδας σχολίων με Gemini: τίτλος + περίληψη 2-3 γραμμών."""
import json
import os
import time

import pandas as pd
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash-lite"
SAMPLE = 14        # σχόλια ανά ομάδα που στέλνονται στο μοντέλο
OUT = "data/cluster_summaries.csv"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

PROMPT = """Σου δίνω δείγμα σχολίων πολιτών από μία ομάδα σχολίων δημόσιας διαβούλευσης
για νομοσχέδιο. Τα σχόλια της ομάδας έχουν κοινό θέμα.
Βασίσου ΜΟΝΟ στα σχόλια που σου δίνω, χωρίς να προσθέσεις πληροφορίες από αλλού.
Γράψε στα ελληνικά:
- "title": σύντομος τίτλος (έως 8 λέξεις) που περιγράφει το κοινό επιχείρημα ή θέμα.
- "summary": περίληψη 2-3 προτάσεων: τι ζητούν ή τι υποστηρίζουν οι πολίτες αυτής της ομάδας.
Απάντησε ΜΟΝΟ με JSON της μορφής {"title": "...", "summary": "..."}.

Σχόλια:
"""


def short(t):
    t = " ".join(str(t).split())
    return t if len(t) <= 600 else t[:450] + " [...] " + t[-150:]


df = pd.read_parquet("data/clustered_comments.parquet")
df = df[df["cluster"] >= 0]

rows = []
for (cid, cl), g in df.groupby(["consultation_id", "cluster"]):
    uniq = g.drop_duplicates("norm")
    sample = uniq.sample(min(SAMPLE, len(uniq)), random_state=1)
    items = "\n".join(f"- {short(t)}" for t in sample["content"])
    result = {"title": "?", "summary": "?"}
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=PROMPT + items,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            result = json.loads(resp.text)
            break
        except Exception as e:
            print(f"  σφάλμα ({str(e)[:100]}). Ξαναδοκιμάζω...")
            time.sleep(15 * (attempt + 1))
    rows.append({"consultation_id": cid, "cluster": cl, "n_comments": len(g),
                 "title": result.get("title", "?"), "summary": result.get("summary", "?")})
    print(f"Διαβούλευση {cid}, ομάδα {cl}: {rows[-1]['title']}")
    time.sleep(4)

pd.DataFrame(rows).to_csv(OUT, index=False)
print("Αποθηκεύτηκε το", OUT)