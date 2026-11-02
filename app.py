"""Εφαρμογή Streamlit: «Τι είπαν οι πολίτες» – ανάλυση σχολίων δημόσιας διαβούλευσης."""
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Τι είπαν οι πολίτες", layout="wide")

# Γραμματοσειρά Arial (έχει έντονο πάχος παντού) και σκούρο χρώμα κειμένου
st.markdown("""
<style>
.stApp, .stApp p, .stApp li, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"] p,
.stApp [data-testid="stCaptionContainer"] p, .stApp [data-baseweb="tab"] p,
.stApp table, .stApp th, .stApp td {
    font-family: Arial, "Segoe UI", Tahoma, sans-serif !important;
    color: #111111 !important;
}
.stApp p, .stApp li, .stApp td { font-size: 1.05rem; line-height: 1.6; font-weight: 400; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp th {
    font-weight: 700 !important; color: #0b2a4a !important;
}
.stApp [data-testid="stMetricValue"] { font-weight: 700 !important; }
.stApp [data-testid="stMetricLabel"] p { font-weight: 700 !important; }
.stApp [data-baseweb="tab"] p { font-weight: 700 !important; font-size: 1.05rem; }
.stApp [data-testid="stCaptionContainer"] p {
    color: #222222 !important; font-size: 0.98rem !important;
}
</style>
""", unsafe_allow_html=True)

CHART_STANCES = ["υπέρ", "κατά", "ουδέτερο"]
STANCE_ORDER = CHART_STANCES + ["?"]
COLOR = {"υπέρ": "green", "κατά": "red", "ουδέτερο": "gray"}


@st.cache_data
def load():
    df = pd.read_parquet("data/final_comments.parquet")
    df["consultation_id"] = df["consultation_id"].astype(str)
    return df


def pct(series, order):
    """Ποσοστά (%) ανά κατηγορία, με σταθερή σειρά."""
    return (series.value_counts(normalize=True).reindex(order).fillna(0) * 100).round(1)


def bar(series):
    """Γράφημα στηλών με μεγάλα, σκούρα γράμματα και την τιμή πάνω από κάθε στήλη."""
    data = series.reset_index()
    data.columns = ["κατηγορία", "pct"]
    base = alt.Chart(data).encode(
        x=alt.X("κατηγορία:N", sort=None, axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y("pct:Q", axis=alt.Axis(title="%")),
    )
    bars = base.mark_bar(color="#1f4e79")
    labels = base.mark_text(dy=-8, fontSize=16, fontWeight="bold", color="#111111").encode(
        text=alt.Text("pct:Q", format=".0f"))
    chart = (bars + labels).properties(height=320).configure_axis(
        labelFontSize=16, titleFontSize=16, labelColor="#111111", titleColor="#111111",
        labelFontWeight="bold",
    )
    st.altair_chart(chart, use_container_width=True)


def clean(text, n=400):
    return " ".join(str(text).split())[:n]


def tag(stance, proposal):
    """Χρωματιστή ετικέτα: στάση + ένδειξη αν περιέχει πρόταση."""
    s = f":{COLOR.get(stance, 'gray')}[**{stance}**]"
    return s + (" · με πρόταση" if proposal == "ναι" else "")


df = load()

st.title("Τι είπαν οι πολίτες")
st.caption("Ανάλυση σχολίων δημόσιας διαβούλευσης (opengov.gr): ομάδες επιχειρημάτων, "
           "στάση και σχόλια που επαναλαμβάνονται.")

titles = df.drop_duplicates("consultation_id").set_index("consultation_id")["consultation_title"]
cid = st.sidebar.selectbox("Διαβούλευση", list(titles.index), format_func=lambda c: titles[c])
d = df[df["consultation_id"] == cid]
st.header(titles[cid])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Σχόλια", f"{len(d):,}")
c2.metric("Μοναδικά σχόλια", f"{d['norm'].nunique():,}")
c3.metric("Σχόλια με περισσότερα αντίγραφα", f"{(d['dup_count'] > 1).mean() * 100:.0f}%")
dd = d[d["stance"] != "σύντομο"]
c4.metric("Περιέχουν πρόταση", f"{(dd['has_proposal'] == 'ναι').mean() * 100:.0f}%")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Στάση και προτάσεις", "Ομάδες επιχειρημάτων", "Σχόλια που επαναλαμβάνονται", "Αναζήτηση"]
)

with tab1:
    left, right = st.columns(2)
    with left:
        st.subheader("Γενική στάση (%)")
        bar(pct(dd["stance"], CHART_STANCES))
    with right:
        st.subheader("Περιέχει πρόταση (%)")
        bar(pct(dd["has_proposal"], ["ναι", "όχι"]))
    st.subheader("Ποσοστό σχολίων με πρόταση, ανά στάση")
    prop = (dd.groupby("stance")["has_proposal"]
              .apply(lambda s: f"{(s == 'ναι').mean() * 100:.1f}%")
              .reindex(CHART_STANCES).dropna().rename("Με πρόταση"))
    prop.index.name = "Στάση"
    st.table(prop)
    st.caption("Οι ετικέτες παράγονται αυτόματα από γλωσσικό μοντέλο (Gemini) και δεν είναι "
               "ελεγμένες σχόλιο προς σχόλιο. Η ακρίβειά τους μετρήθηκε σε δείγμα (βλ. αναφορά). "
               "«Κατά» σημαίνει κυρίως επικριτικό σχόλιο, ακόμη κι αν περιέχει προτάσεις.")

with tab2:
    groups = (
        d[d["cluster"] >= 0]
        .groupby(["cluster", "cluster_title", "cluster_summary"])
        .size().reset_index(name="n").sort_values("n", ascending=False)
    )
    for r in groups.itertuples():
        with st.expander(f"{r.cluster_title} — {r.n} σχόλια"):
            st.write(r.cluster_summary)
            g = d[d["cluster"] == r.cluster]
            st.caption("Στάση: " + " · ".join(
                f"{k} {v:.0f}%" for k, v in pct(g["stance"], STANCE_ORDER).items() if v > 0))
            st.markdown("**Δείγμα σχολίων:**")
            uniq = g.drop_duplicates("norm")
            for r2 in uniq.sample(min(3, len(uniq)), random_state=1).itertuples():
                st.markdown(tag(r2.stance, r2.has_proposal))
                st.write(clean(r2.content))

with tab3:
    rep = (d.drop_duplicates("norm")
             .query("dup_count > 1")
             .sort_values("dup_count", ascending=False).head(10))
    if rep.empty:
        st.info("Δεν βρέθηκαν σχόλια που επαναλαμβάνονται αυτούσια.")
    for r in rep.itertuples():
        st.markdown(f"**{r.dup_count} φορές** · " + tag(r.stance, r.has_proposal))
        st.write(clean(r.content, 500))
        st.divider()

with tab4:
    q = st.text_input("Αναζήτηση λέξης ή φράσης στα σχόλια")
    col_a, col_b = st.columns(2)
    stance = col_a.selectbox("Στάση", ["όλες"] + CHART_STANCES)
    prop_f = col_b.selectbox("Πρόταση", ["όλα", "με πρόταση", "χωρίς πρόταση"])
    res = d
    if q:
        res = res[res["content"].str.contains(q, case=False, na=False, regex=False)]
    if stance != "όλες":
        res = res[res["stance"] == stance]
    if prop_f == "με πρόταση":
        res = res[res["has_proposal"] == "ναι"]
    elif prop_f == "χωρίς πρόταση":
        res = res[res["has_proposal"] == "όχι"]
    st.write(f"{len(res)} σχόλια")
    for r in res.drop_duplicates("norm").head(20).itertuples():
        st.markdown(tag(r.stance, r.has_proposal))
        st.write(clean(r.content))