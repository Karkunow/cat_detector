import pandas as pd
import streamlit as st

from storage import DB_PATH, init_db

CAT_COLORS = {"white": "#e8e8e8", "black": "#3a3a3a", "unknown": "#f2a900"}

st.set_page_config(page_title="Кошачий туалет", page_icon="🐾")
st.title("🐾 Статистика відвідувань лотка")

conn = init_db(DB_PATH)
df = pd.read_sql("SELECT * FROM events", conn, parse_dates=["timestamp"])

if df.empty:
    st.info("Ще немає подій у events.db — запусти `python classify_offline.py run`.")
    st.stop()

df["date"] = df["timestamp"].dt.date
df["day_night"] = df["is_day"].map({1: "день", 0: "ніч"})

visits = df[df["cat"].isin(["white", "black"])]

col1, col2, col3 = st.columns(3)
col1.metric("Візитів всього", len(visits))
col2.metric("Білий кіт", (visits["cat"] == "white").sum())
col3.metric("Чорний кіт", (visits["cat"] == "black").sum())

st.subheader("Візити на день")
if not visits.empty:
    daily = visits.groupby(["date", "cat"]).size().unstack(fill_value=0)
    st.bar_chart(daily, color=[CAT_COLORS.get(c, "#888888") for c in daily.columns])
else:
    st.info("Поки що жодного підтвердженого візиту (тільки passby/unknown).")

st.subheader("День vs ніч")
if not visits.empty:
    dn = visits.groupby(["day_night", "cat"]).size().unstack(fill_value=0)
    st.bar_chart(dn)

st.subheader("Останні події")
show_cols = ["timestamp", "cat", "day_night", "confidence", "dwell_seconds", "source_clip"]
st.dataframe(df.sort_values("timestamp", ascending=False)[show_cols], hide_index=True)
