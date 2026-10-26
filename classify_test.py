"""Δοκιμή ταξινόμησης σχολίων (υπέρ / κατά / πρόταση / άλλο) με Gemini, σε 30 σχόλια."""
import json
import os

import pandas as pd
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash-lite"  # αν δώσει σφάλμα, δοκίμασε gemini-3.1-flash-lite
N = 30

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

PROMPT = """Είσαι αναλυτής δημόσιων διαβουλεύσεων. Σου δίνω σχόλια πολιτών σε
διαβούλευση για νομοσχέδιο. Για κάθε σχόλιο διάλεξε ΜΙΑ ετικέτα:

- "υπέρ": το σχόλιο στηρίζει συνολικά τη ρύθμιση.
- "κατά": το σχόλιο διαφωνεί με τη ρύθμιση ή ζητά να αποσυρθεί ή να αλλάξει ριζικά.
- "πρόταση": το κύριο περιεχόμενο είναι μια συγκεκριμένη πρόταση αλλαγής,
  προσθήκης ή διόρθωσης σε άρθρο (ακόμη κι αν συμφωνεί γενικά).
- "άλλο": ερώτηση, ουδέτερο ή τεχνικό σχόλιο, άσχετο κείμενο, ή δεν μπορείς να καταλάβεις.

Απάντησε ΜΟΝΟ με JSON, λίστα της μορφής:
[{"id": 0, "label": "υπέρ"}, {"id": 1, "label": "κατά"}, ...]

Σχόλια:
"""

df = pd.read_parquet("data/clustered_comments.parquet")
df = df[df["norm"].str.len() >= 20].drop_duplicates(["consultation_id", "norm"])
sample = df.sample(N, random_state=1).reset_index(drop=True)

items = "\n".join(
    f"[{i}] {text[:700].replace(chr(10), ' ')}" for i, text in enumerate(sample["content"])
)

resp = client.models.generate_content(
    model=MODEL,
    contents=PROMPT + items,
    config=types.GenerateContentConfig(response_mime_type="application/json"),
)
labels = {d["id"]: d["label"] for d in json.loads(resp.text)}
sample["llm_label"] = [labels.get(i, "?") for i in range(len(sample))]

print(sample["llm_label"].value_counts())
sample[["consultation_id", "comment_row_id", "content", "llm_label"]].to_csv(
    "data/classify_test.csv", index=False
)
print("Αποθηκεύτηκε το data/classify_test.csv")