import streamlit as st


st.set_page_config(
    page_title="Hindcon TDS Assistant",
    page_icon="📘",
    layout="wide"
)


st.title("📘 Hindcon TDS Product Assistant")

st.write(
    "Ask questions about Hindcon products using their "
    "Technical Data Sheets (TDS)."
)


user_query = st.chat_input(
    "Ask about a Hindcon product..."
)


if user_query:
    st.write("You asked:")
    st.write(user_query)