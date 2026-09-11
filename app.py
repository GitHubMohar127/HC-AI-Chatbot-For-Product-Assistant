import streamlit as st

from src.answer_generator import (
    answer_from_tds
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title=
        "Hindcon TDS Assistant",

    page_icon=
        "📘",

    layout=
        "wide"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📘 Hindcon TDS Product Assistant"
)

st.write(
    "Ask questions about Hindcon products "
    "using their Technical Data Sheets (TDS)."
)


# ============================================================
# CONVERSATION STATE
# ============================================================

if "conversation_state" not in st.session_state:

    st.session_state.conversation_state = {

        "product":
            None,

        "product_id":
            None,

        "product_name":
            None,

        "category":
            None
    }


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY OLD MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================

user_query = st.chat_input(
    "Ask about a Hindcon product..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if user_query:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(

        {

            "role":
                "user",

            "content":
                user_query
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_query
        )


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching the TDS..."
        ):

            try:

                result = (
                    answer_from_tds(

                        query=
                            user_query,

                        state=
                            st.session_state[
                                "conversation_state"
                            ],

                        k=5
                    )
                )


                # ------------------------------------------------
                # ANSWER
                # ------------------------------------------------

                answer = (
                    result["answer"]
                )

                st.markdown(
                    answer
                )


                # ------------------------------------------------
                # STATUS
                # ------------------------------------------------

                status = (
                    result.get(
                        "status"
                    )
                )

                if status:

                    st.caption(
                        f"Status: {status}"
                    )


                # ------------------------------------------------
                # SOURCES
                # ------------------------------------------------

                sources = (
                    result.get(
                        "sources",
                        []
                    )
                )

                if sources:

                    with st.expander(
                        "📚 TDS Sources"
                    ):

                        for source in sources:

                            st.write(

                                f"**Product:** "
                                f"{source.get('product_name', 'N/A')}"
                            )

                            # st.write(

                            #     f"**Product ID:** "
                            #     f"{source.get('product_id', 'N/A')}"
                            # )

                            # st.write(

                            #     f"**Section:** "
                            #     f"{source.get('section', 'N/A')}"
                            # )

                            # st.write(

                            #     f"**Document:** "
                            #     f"{source.get('document_name', 'N/A')}"
                            # )

                            tds_link = (
                                source.get(
                                    "tds_link"
                                )
                            )

                            if tds_link:

                                st.markdown(

                                    f"[Open TDS Document]"
                                    f"({tds_link})"
                                )

                            st.divider()


                # ------------------------------------------------
                # EVIDENCE SCORE
                # ------------------------------------------------

                evidence_score = (
                    result.get(
                        "evidence_score"
                    )
                )

                if evidence_score is not None:

                    st.caption(

                        f"Evidence score: "
                        f"{evidence_score}"
                    )


                # ------------------------------------------------
                # SAVE ASSISTANT MESSAGEs
                # ------------------------------------------------

                st.session_state.messages.append(

                    {

                        "role":
                            "assistant",

                        "content":
                            answer
                    }
                )


            except Exception as e:

                st.error(
                    "An error occurred while "
                    "processing your question."
                )

                st.exception(e)