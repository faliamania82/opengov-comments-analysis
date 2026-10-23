"""Απλή προβολή στατιστικών και σχολίων με Streamlit."""

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st


STATS_PATH = Path("stats.csv")
DATABASE_PATH = Path("opengov.duckdb")

st.set_page_config(page_title="Σχόλια δημόσιας διαβούλευσης", layout="wide")
st.title("Εξερεύνηση δημόσιων διαβουλεύσεων")

if not STATS_PATH.exists() or not DATABASE_PATH.exists():
    st.error("Δεν βρέθηκαν stats.csv και opengov.duckdb. Εκτελέστε πρώτα: python explore.py")
    st.stop()

stats = pd.read_csv(STATS_PATH)
st.dataframe(stats, use_container_width=True, hide_index=True)

if stats.empty:
    st.info("Δεν υπάρχουν διαθέσιμα στατιστικά.")
    st.stop()

key_to_title = dict(zip(stats["consultation_key"].astype(str), stats["consultation_title"].astype(str)))
selected_key = st.selectbox(
    "Διαβούλευση",
    options=list(key_to_title),
    format_func=lambda key: f"{key_to_title[key]} [{key}]" if key_to_title[key] != key else key,
)

connection = duckdb.connect(str(DATABASE_PATH), read_only=True)
comments = connection.execute(
    """SELECT comment_text
       FROM comments
       WHERE consultation_key = ?
       ORDER BY random()
       LIMIT 10""",
    [selected_key],
).df()
connection.close()

st.subheader(f"Τυχαία σχόλια: {key_to_title[selected_key]}")
if comments.empty:
    st.info("Δεν βρέθηκαν σχόλια για αυτή τη διαβούλευση.")
else:
    for index, comment in enumerate(comments["comment_text"], start=1):
        st.markdown(f"**{index}.** {comment if comment.strip() else '_(κενό σχόλιο)_'}")