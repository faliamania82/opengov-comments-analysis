# Repository instructions

## Run the project

The documented setup and initial dataset explorer are in [README.md](../README.md):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python explore.py
streamlit run app_explore.py
```

Set `HF_TOKEN` in the environment if the Hugging Face dataset requires access. The later analysis workflow uses the selected comments produced by `explore.py`:

```bash
python cluster.py
export GEMINI_API_KEY=...
python classify_v3.py
python summarize.py
python build_final.py
streamlit run app.py
```

`classify_v3.py` and `summarize.py` call Gemini. The classifier appends results and skips comment IDs already present in its output, so it can resume after interruption. `python evaluate.py` compares the existing manual and model-labeled evaluation samples; it expects `data/eval_key.csv` and `data/eval_sample.xlsx`.

There is no configured build, test, or lint runner. `requirements.txt` covers the initial explorer, but later scripts also import `duckdb`, `huggingface_hub`, `sentence-transformers`, `scikit-learn`, `google-genai`, and `openpyxl`.

## Architecture and data flow

- This is a script-based Python project rather than an installable package. `explore.py` downloads and flattens the nested Hugging Face dataset, writes `stats.csv`, and selects consultation rows into `data/selected_comments.parquet`. `app_explore.py` is the separate, earlier explorer that reads those stats and `opengov.duckdb`.
- The analysis path starts at `cluster.py`: it normalizes text, counts repeated comments within each consultation, excludes comments shorter than 20 normalized characters from clustering, and writes `data/clustered_comments.parquet`.
- `classify_v3.py` classifies unique non-short comments; `summarize.py` generates titles and summaries for clusters. Both use Gemini. `build_final.py` joins these results back onto all comments and creates `data/final_comments.parquet`; the main dashboard in `app.py` reads that artifact.
- The distinction between scripts named `classify_all.py`, `classify_v2.py`, and `classify_v3.py` matters: the current final-data build consumes the v3 output. The current definitions of stance and proposal are in `classify_v3.py`; `prompts.md` records historical iterations and evaluation context.

## Data and labeling conventions

- Treat `consultation_id` and `comment_row_id` as strings before joining CSV and Parquet data. Comment uniqueness and duplicate propagation are scoped to `(consultation_id, norm)`, not text across the whole dataset.
- `cluster.py` defines `norm` by lowercasing, removing accents and punctuation, and collapsing whitespace. Reuse this normalization and the consultation scope when changing duplicate handling or merging model labels.
- Keep category values compatible with `classify_v3.py` and `app.py`: stance is `υπέρ`, `κατά`, or `ουδέτερο`; proposal is `ναι` or `όχι`. Short comments are represented as `σύντομο` in the final data, and unclustered comments use cluster `-1`.
- Prompts and user-facing labels are in Greek. Changes to model prompts can change dashboard aggregates, so keep the definitions aligned with the evaluation scripts and `prompts.md`.
