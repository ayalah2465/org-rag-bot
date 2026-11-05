"""ממשק המשתמש - Streamlit."""

import streamlit as st

from config import (
    EXPANSION_ENABLED_BY_DEFAULT,
    HYBRID_ENABLED_BY_DEFAULT,
    TOP_K,
)
from query import answer

st.set_page_config(page_title="בוט ידע ארגוני - תחנות טיפת חלב", page_icon="", layout="wide")

st.title("בוט ידע ארגוני - תחנות טיפת חלב, מרחב ירושלים")
st.caption("הבוט עונה אך ורק על סמך מסמכי הארגון: נוהל זימון תורים, נוהל רישום ומסמכים, נוהל תפעול תחנות.")

with st.sidebar:
    st.header("הגדרות אחזור")
    top_k = st.slider("מספר קטעים שיוחזרו (Top-K)", 1, 10, TOP_K)
    use_hybrid = st.toggle(
        "חיפוש היברידי (וקטורי + BM25)",
        value=HYBRID_ENABLED_BY_DEFAULT,
        help="מיזוג תוצאות וקטוריות עם התאמה מילולית. ללא קריאת LLM.",
    )
    use_expansion = st.toggle(
        "הרחבת שאלה (Query Expansion)",
        value=EXPANSION_ENABLED_BY_DEFAULT,
        help="קריאת LLM נוספת לניסוח מחדש של השאלה. מוסיף השהיה ועלות קטנה.",
    )
    show_sources = st.toggle("הצגת המקורות", value=True)
    if st.button("ניקוי השיחה"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            st.caption("מקורות: " + " · ".join(msg["sources"]))
        if msg.get("details"):
            with st.expander("קטעים שנשלחו למודל"):
                st.write(msg["details"])

question = st.chat_input("שאלו שאלה על נהלי התחנות...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("מחפש במסמכי הארגון..."):
            try:
                result = answer(
                    question,
                    top_k=top_k,
                    use_hybrid=use_hybrid,
                    use_expansion=use_expansion,
                )
            except Exception as exc:
                st.error("שגיאה בהרצת השאילתה: " + str(exc))
                st.stop()

        st.markdown(result["answer"])

        details = ""
        if result["sources"] and show_sources:
            st.caption("מקורות: " + " · ".join(result["sources"]))
        if result["effective_query"] != question:
            st.caption("השאילתה שהופעלה בפועל: " + result["effective_query"])

        if result["nodes"]:
            details = "\n\n".join(
                "**קטע " + str(i) + "** (ציון " + f"{n.score:.3f}" + ")\n\n" + n.get_text()[:600]
                for i, n in enumerate(result["nodes"], 1)
                if n.score is not None
            )
            with st.expander("קטעים שנשלחו למודל"):
                for i, n in enumerate(result["nodes"], 1):
                    st.markdown("**קטע " + str(i) + "** | ציון: " + f"{n.score:.3f}")
                    st.text(n.get_text()[:800])
                    st.divider()

        st.caption("זמן ריצה: " + f"{result['elapsed']:.2f}" + " שניות")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "details": details,
    })
