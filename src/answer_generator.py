import os

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

from langchain_core.prompts import (
    ChatPromptTemplate
)

from src.retriever import (
    retrieve_with_validation_context
)

from src.query_router import (
    QUERY_STATUS
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:

    raise ValueError(
        "GEMINI_API_KEY not found in .env file."
    )


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(

    model=
        "gemini-3.1-flash-lite",

    api_key=
        GEMINI_API_KEY
)


# ============================================================
# TDS ANSWER PROMPT
# ============================================================

answer_prompt = ChatPromptTemplate.from_messages(

    [

        (

            "system",

            """
You are a Technical Data Sheet (TDS) assistant
for Hindcon products.

Your job is to answer questions ONLY using
the provided TDS context.

STRICT RULES:

1. Use only information present in the TDS context.

2. Never use outside knowledge.

3. Never guess or invent missing information.

4. If the requested information is not present
   in the TDS context, say:

   "The requested information is not available
   in the provided TDS."

5. Preserve numerical values exactly as provided
   in the TDS.

6. Preserve units exactly as provided.

7. Do not change or calculate values unless the
   TDS itself provides the calculation.

8. If multiple values exist, clearly explain
   what each value represents.

9. Prefer dedicated technical specifications
   over unrelated mentions of the same word.

10. Give a concise and clear answer.

TDS CONTEXT:

{context}
"""

        ),

        (

            "human",

            "{question}"

        )
    ]
)


# ============================================================
# FORMAT TDS CONTEXT
# ============================================================

def format_tds_context(
    documents
):

    context_parts = []

    for index, doc in enumerate(
        documents,
        start=1
    ):

        metadata = (
            doc.metadata
        )

        product_name = (
            metadata.get(
                "product_name",
                "Unknown"
            )
        )

        product_id = (
            metadata.get(
                "product_id",
                "Unknown"
            )
        )

        section = (
            metadata.get(
                "section",
                "Unknown"
            )
        )

        document_name = (
            metadata.get(
                "document_name",
                "Unknown"
            )
        )

        content = (
            doc.page_content
        )

        context_parts.append(

            f"""
--- TDS SOURCE {index} ---

Product:
{product_name}

Product ID:
{product_id}

Section:
{section}

Document:
{document_name}

Content:
{content}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# FORMAT SOURCES
# ============================================================

def format_sources(documents):
    """
    Return unique TDS sources.

    Multiple retrieved chunks can belong to the same PDF,
    so only one source entry is shown for each unique TDS.
    """

    sources = []
    seen_sources = set()

    for doc in documents:

        metadata = doc.metadata

        product_id = metadata.get("product_id", "")
        product_name = metadata.get("product_name", "")
        document_name = metadata.get("document_name", "")
        tds_link = metadata.get("tds_link", "")
        section = metadata.get("section", "")

        # Use PDF link as the primary unique identifier.
        # If link is unavailable, fall back to product + document name.
        source_key = (
            tds_link
            if tds_link
            else f"{product_id}_{document_name}"
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)

        sources.append({
            "product_name": product_name,
            "product_id": product_id,
            "document_name": document_name,
            "tds_link": tds_link,
            "section": section
        })

    return sources


# ============================================================
# ANSWER FROM TDS
# ============================================================

def answer_from_tds(
    query: str,
    state: dict,
    k: int = 5
):

    retrieval = (
        retrieve_with_validation_context(
            query,
            state,
            k=k
        )
    )

    # --------------------------------------------------------
    # INVALID / INCOMPLETE / NOT FOUND
    # --------------------------------------------------------

    if (
        retrieval["status"]
        != QUERY_STATUS["VALID"]
    ):

        return {

            "answer":
                retrieval["message"],

            "status":
                retrieval["status"],

            "sources":
                [],

            "evidence_score":
                retrieval.get(
                    "evidence",
                    {}
                ).get(
                    "score",
                    0
                )
        }


    # --------------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------------

    documents = (
        retrieval[
            "documents"
        ][:3]
    )

    if not documents:

        return {

            "answer":
                (
                    "The requested information "
                    "is not available in the "
                    "provided TDS."
                ),

            "status":
                QUERY_STATUS[
                    "INFORMATION_NOT_AVAILABLE"
                ],

            "sources":
                [],

            "evidence_score":
                0
        }


    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context = (
        format_tds_context(
            documents
        )
    )


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    messages = (
        answer_prompt.format_messages(

            question=query,

            context=context
        )
    )


    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    response = llm.invoke(
        messages
    )


    answer_text = (
        response.text
    )


    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    sources = (
        format_sources(
            documents
        )
    )


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "answer":
            answer_text,

        "status":
            retrieval["status"],

        "sources":
            sources,

        "evidence_score":
            retrieval[
                "evidence"
            ]["score"]
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Loading TDS chunks..."
    )

    from src.retriever import (
        FINAL_CHUNKS
    )

    print(
        f"Loaded {len(FINAL_CHUNKS)} TDS chunks."
    )

    print(
        "\nTesting Answer Generator..."
    )

    state = {

        "product":
            None,

        "product_id":
            None,

        "product_name":
            None,

        "category":
            None
    }

    result = answer_from_tds(

        query=
            "What is the setting time of Hind Plug - S?",

        state=
            state,

        k=5
    )

    print(
        "\nANSWER:"
    )

    print(
        result["answer"]
    )

    print(
        "\nSTATUS:"
    )

    print(
        result["status"]
    )

    print(
        "\nSOURCES:"
    )

    for source in result[
        "sources"
    ]:

        print(
            source
        )

    print(
        "\nEVIDENCE SCORE:"
    )

    print(
        result[
            "evidence_score"
        ]
    )