# Ανάλυση σχολίων opengov.gr

Πρώτο στάδιο εξερεύνησης του Hugging Face dataset `glossAPI/opengov-deliberations-v2`. Δεν εκτελεί clustering ή ανάλυση με LLM.

## Εγκατάσταση

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Εκτέλεση

```bash
python explore.py
streamlit run app_explore.py
```

Το `explore.py` εξετάζει configs/splits σε streaming, τυπώνει schema και δείγματα, και δημιουργεί `stats.csv` μαζί με `opengov.duckdb`. Το DuckDB κρατά τα σχόλια για την επιλογή τυχαίου δείγματος από το Streamlit. Αν το dataset απαιτεί πρόσβαση, ορίστε το `HF_TOKEN` στο περιβάλλον πριν από την εκτέλεση.

Το `duplicate_pct` μετρά το ποσοστό σχολίων που ανήκουν σε ομάδα με τουλάχιστον δύο ίδια κανονικοποιημένα κείμενα. Τα κενά κείμενα εξαιρούνται από αυτό το μέτρο. Οι σύντομες/κενές απαντήσεις μετρώνται ως κείμενα κάτω από 20 χαρακτήρες μετά από αφαίρεση περιφερειακών κενών.