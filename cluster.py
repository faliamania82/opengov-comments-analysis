"""Ομαδοποίηση των σχολίων κάθε διαβούλευσης με embeddings + KMeans.

Βήματα: καθαρισμός κειμένου, εντοπισμός σχολίων που επαναλαμβάνονται αυτούσια,
embeddings με πολύγλωσσο μοντέλο, ομαδοποίηση, αποθήκευση αποτελεσμάτων.
"""
import re
import unicodedata

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

MODEL = "intfloat/multilingual-e5-small"
K = {"13": 10, "8": 10, "11": 8}  # αριθμός ομάδων ανά διαβούλευση
MIN_CHARS = 20  # πιο σύντομα σχόλια δεν ομαδοποιούνται


def normalize(text):
    """Πεζά, χωρίς τόνους και σημεία στίξης, ώστε να εντοπίζονται ίδια σχόλια."""
    t = unicodedata.normalize("NFD", str(text).lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    df = pd.read_parquet("data/selected_comments.parquet")
    df["content"] = df["content"].fillna("").astype(str)
    df["norm"] = df["content"].map(normalize)

    # Πόσες φορές εμφανίζεται το ίδιο σχόλιο μέσα στην ίδια διαβούλευση
    df["dup_count"] = df.groupby(["consultation_id", "norm"])["norm"].transform("size")
    df["short"] = df["norm"].str.len() < MIN_CHARS
    df["cluster"] = -1  # -1 = πολύ σύντομο σχόλιο, δεν ομαδοποιείται

    model = SentenceTransformer(MODEL)
    model.max_seq_length = 256

    for cid, k in K.items():
        mask = (df["consultation_id"] == cid) & ~df["short"]
        unique = df[mask].drop_duplicates("norm")
        print(f"\nΔιαβούλευση {cid}: {mask.sum()} σχόλια, {len(unique)} μοναδικά")

        # Το e5 θέλει το πρόθεμα "query: " σε κάθε κείμενο
        texts = ["query: " + t for t in unique["content"]]
        emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=True)

        labels = KMeans(n_clusters=min(k, len(unique)), n_init=10,
                        random_state=42).fit_predict(emb)
        label_map = dict(zip(unique["norm"], labels))
        df.loc[mask, "cluster"] = df.loc[mask, "norm"].map(label_map).astype(int)

        # Έλεγχος με το μάτι: μέγεθος και 2 δείγματα ανά ομάδα
        for c in sorted(set(labels)):
            members = df[mask & (df["cluster"] == c)]
            print(f"  Ομάδα {c}: {len(members)} σχόλια")
            for s in members["content"].head(2):
                print("     -", s[:120].replace("\n", " "))

    df.to_parquet("data/clustered_comments.parquet", index=False)
    print("\nΑποθηκεύτηκε το data/clustered_comments.parquet")


if __name__ == "__main__":
    main()