import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.query_router import (
    normalize_text,
    normalize_product_query,
    route_query_with_context,
    QUERY_STATUS,
    RESPONSE_MESSAGES
)


# ============================================================
# PROJECT PATHS
# ============================================================

from pathlib import Path

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

VECTORSTORE_DIR = (
    PROJECT_ROOT / "vectorstore"
)


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name=
        "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CHROMA
# ============================================================

vectorstore = Chroma(
    collection_name=
        "hindcon_tds_local",

    embedding_function=
        embeddings,

    persist_directory=
        str(VECTORSTORE_DIR)
)


# ============================================================
# LOAD ALL TDS CHUNKS
# ============================================================

def load_all_chunks():

    data = vectorstore.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = []

    raw_documents = (
        data.get(
            "documents",
            []
        )
    )

    raw_metadatas = (
        data.get(
            "metadatas",
            []
        )
    )

    for text, metadata in zip(
        raw_documents,
        raw_metadatas
    ):

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata=metadata or {}
            )
        )

    return documents


FINAL_CHUNKS = load_all_chunks()


# ============================================================
# QUERY TERMS
# ============================================================

def extract_query_terms(
    query: str
):

    normalized_query = (
        normalize_text(
            query
        )
    )

    stop_words = {

        "what",
        "is",
        "the",
        "of",
        "for",
        "how",
        "does",
        "do",
        "tell",
        "me",
        "about",
        "please",
        "can",
        "you",
        "give",
        "which",
        "a",
        "an",
        "and",
        "or",
        "to",
        "in",
        "on",
        "at"
    }

    words = [

        word

        for word
        in normalized_query.split()

        if word
        not in stop_words
    ]

    terms = words.copy()

    for i in range(
        len(words) - 1
    ):

        terms.append(
            f"{words[i]} {words[i + 1]}"
        )

    return terms


# ============================================================
# TECHNICAL FIELD ALIASES
# ============================================================

TECHNICAL_FIELD_ALIASES = {

    "SETTING TIME": [

        "setting time",
        "set time",
        "initial setting",
        "final setting",
        "setting at"
    ],

    "COMPRESSIVE STRENGTH": [

        "compressive strength",
        "compressive"
    ],

    "COVERAGE": [

        "coverage",
        "consumption",
        "application rate",
        "material required"
    ],

    "SHELF LIFE": [

        "shelf life",
        "storage life",
        "expiry",
        "expiration"
    ],

    "PACKING": [

        "packing",
        "pack size",
        "package",
        "packaging"
    ],

    "DOSAGE": [

        "dosage",
        "dose",
        "recommended dosage"
    ],

    "DENSITY": [

        "density",
        "bulk density"
    ],

    "PARTICLE SIZE": [

        "particle size",
        "particle"
    ]
}


# ============================================================
# TECHNICAL FIELD DETECTION
# ============================================================

def detect_technical_field(
    query: str
):

    normalized_query = (
        normalize_text(
            query
        )
    )

    matches = []

    for (
        field,
        aliases
    ) in TECHNICAL_FIELD_ALIASES.items():

        for alias in aliases:

            if normalize_text(
                alias
            ) in normalized_query:

                matches.append(
                    field
                )

                break

    return list(
        dict.fromkeys(
            matches
        )
    )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def search_tds(
    query: str,
    k: int = 5,
    product_id=None,
    section=None
):

    if product_id and section:

        return vectorstore.similarity_search(
            query,
            k=k,
            filter={
                "$and": [
                    {
                        "product_id":
                            product_id
                    },
                    {
                        "section":
                            section
                    }
                ]
            }
        )

    if product_id:

        return vectorstore.similarity_search(
            query,
            k=k,
            filter={
                "product_id":
                    product_id
            }
        )

    if section:

        return vectorstore.similarity_search(
            query,
            k=k,
            filter={
                "section":
                    section
            }
        )

    return vectorstore.similarity_search(
        query,
        k=k
    )


# ============================================================
# KEYWORD SEARCH
# ============================================================

def keyword_search(
    query,
    documents,
    k=5,
    product_id=None,
    section=None
):

    query_terms = (
        extract_query_terms(
            query
        )
    )

    scored_documents = []

    for doc in documents:

        if (
            product_id
            and doc.metadata.get(
                "product_id"
            )
            != product_id
        ):
            continue

        if (
            section
            and doc.metadata.get(
                "section"
            )
            != section
        ):
            continue

        text = normalize_text(
            doc.page_content
        )

        if not text:
            continue

        score = 0

        for term in query_terms:

            if " " in term:

                if term in text:
                    score += 3

            else:

                if term in text:
                    score += 1

        if score > 0:

            scored_documents.append(
                (
                    score,
                    doc
                )
            )

    scored_documents.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        doc

        for score, doc
        in scored_documents[:k]
    ]


# ============================================================
# TECHNICAL FIELD SEARCH
# ============================================================

def technical_field_search(
    query,
    documents,
    k=5,
    product_id=None
):

    technical_fields = (
        detect_technical_field(
            query
        )
    )

    if not technical_fields:
        return []

    scored_documents = []

    for doc in documents:

        if product_id:

            if (
                doc.metadata.get(
                    "product_id"
                )
                != product_id
            ):
                continue

        text = normalize_text(
            doc.page_content
        )

        if not text:
            continue

        score = 0

        for field in technical_fields:

            aliases = (
                TECHNICAL_FIELD_ALIASES[
                    field
                ]
            )

            for alias in aliases:

                if normalize_text(
                    alias
                ) in text:

                    score += 20

                    break

        if score > 0:

            scored_documents.append(
                (
                    score,
                    doc
                )
            )

    scored_documents.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        doc

        for score, doc
        in scored_documents[:k]
    ]


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicate_chunks(
    documents
):

    unique_documents = []

    seen_content = set()

    for doc in documents:

        content = normalize_text(
            doc.page_content
        )

        if not content:
            continue

        content_key = re.sub(
            r"\s+",
            " ",
            content
        ).strip()

        if content_key in seen_content:
            continue

        seen_content.add(
            content_key
        )

        unique_documents.append(
            doc
        )

    return unique_documents


# ============================================================
# EVIDENCE SCORE
# ============================================================

def evidence_score(
    query: str,
    document
):

    text = normalize_text(
        document.page_content
    )

    if not text:
        return 0

    score = 0

    normalized_query = (
        normalize_product_query(
            query
        )
    )

    product_name = (
        normalize_product_query(
            str(
                document.metadata.get(
                    "product_name",
                    ""
                )
            )
        )
    )

    # Product match
    if (
        product_name
        and product_name
        in normalized_query
    ):

        score += 5

    # Technical field
    technical_fields = (
        detect_technical_field(
            query
        )
    )

    for field in technical_fields:

        aliases = (
            TECHNICAL_FIELD_ALIASES[
                field
            ]
        )

        for alias in aliases:

            if normalize_text(
                alias
            ) in text:

                score += 30

                break

    # Section relevance
    section = (
        document.metadata.get(
            "section"
        )
    )

    technical_sections = {

        "PROPERTIES",
        "TECHNICAL DATA",
        "SPECIFICATIONS",
        "COVERAGE",
        "SHELF LIFE",
        "PACKING"
    }

    if section in technical_sections:

        score += 10

    # Irrelevant sections
    irrelevant_sections = {

        "GENERAL",
        "DESCRIPTION",
        "HANDLING PRECAUTION",
        "HANDLING PRECAUTIONS"
    }

    if section in irrelevant_sections:

        score -= 20

    # Query terms
    query_terms = (
        extract_query_terms(
            query
        )
    )

    for term in query_terms:

        if term in text:

            if " " in term:
                score += 3
            else:
                score += 1

    # Setting time preference
    if (
        "setting time"
        in normalize_text(query)
    ):

        if "setting at" in text:
            score += 40

        if "setting time" in text:
            score += 40

    return score


# ============================================================
# RANK EVIDENCE
# ============================================================

def rank_evidence(
    query: str,
    documents: list
):

    scored_documents = []

    for doc in documents:

        score = evidence_score(
            query,
            doc
        )

        scored_documents.append(
            (
                score,
                doc
            )
        )

    scored_documents.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        doc

        for score, doc
        in scored_documents
    ]


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    k=5,
    product_id=None,
    section=None
):

    semantic_results = (
        search_tds(
            query,
            k=k,
            product_id=product_id,
            section=section
        )
    )

    keyword_results = (
        keyword_search(
            query,
            FINAL_CHUNKS,
            k=k,
            product_id=product_id,
            section=section
        )
    )

    technical_results = (
        technical_field_search(
            query,
            FINAL_CHUNKS,
            k=k,
            product_id=product_id
        )
    )

    combined = (
        technical_results
        + keyword_results
        + semantic_results
    )

    combined = (
        remove_duplicate_chunks(
            combined
        )
    )

    ranked = rank_evidence(
        query,
        combined
    )

    return ranked[:k]


# ============================================================
# EVIDENCE CHECK
# ============================================================

def check_evidence(
    query: str,
    documents: list
):

    if not documents:

        return {

            "sufficient":
                False,

            "score":
                0,

            "reason":
                "NO_DOCUMENTS"
        }

    ranked_documents = (
        rank_evidence(
            query,
            documents
        )
    )

    best_document = (
        ranked_documents[0]
    )

    score = evidence_score(
        query,
        best_document
    )

    return {

        "sufficient":
            score >= 3,

        "score":
            score,

        "reason":
            (
                "SUFFICIENT"
                if score >= 3
                else "WEAK_EVIDENCE"
            ),

        "best_document":
            best_document
    }


# ============================================================
# CONTEXTUAL QUERY
# ============================================================

def build_contextual_query(
    query: str,
    product: dict
):

    product_name = (
        product["product_name"]
    )

    normalized_query = (
        normalize_text(
            query
        )
    )

    words = set(
        normalized_query.split()
    )

    contains_reference = (

        "it" in words

        or "its" in words

        or "this product"
        in normalized_query

        or "that product"
        in normalized_query

        or "the product"
        in normalized_query
    )

    if contains_reference:

        return (
            f"{query} "
            f"for {product_name}"
        )

    return query


# ============================================================
# CONTEXT-AWARE RETRIEVAL
# ============================================================

def retrieve_with_validation_context(
    query: str,
    state: dict,
    k: int = 5
):

    routing = (
        route_query_with_context(
            query,
            state
        )
    )

    if (
        routing["status"]
        != QUERY_STATUS["VALID"]
    ):

        return {

            "status":
                routing["status"],

            "message":
                routing["message"],

            "documents":
                [],

            "evidence":
                {
                    "score": 0
                }
        }

    product = (
        routing["product"]
    )

    product_id = (
        product["product_id"]
    )

    contextual_query = (
        build_contextual_query(
            query,
            product
        )
    )

    section_detection = (
        routing[
            "section_detection"
        ]
    )

    section = None

    if (
        section_detection["status"]
        == "FOUND"
    ):

        section = (
            section_detection[
                "section"
            ]
        )

    documents = (
        hybrid_search(
            contextual_query,
            k=k,
            product_id=product_id,
            section=section
        )
    )

    evidence = (
        check_evidence(
            contextual_query,
            documents
        )
    )

    if not evidence[
        "sufficient"
    ]:

        return {

            "status":
                QUERY_STATUS[
                    "INFORMATION_NOT_AVAILABLE"
                ],

            "message":
                RESPONSE_MESSAGES[
                    "INFORMATION_NOT_AVAILABLE"
                ],

            "documents":
                [],

            "evidence":
                evidence
        }

    return {

        "status":
            QUERY_STATUS["VALID"],

        "message":
            "Query is valid.",

        "documents":
            documents,

        "evidence":
            evidence,

        "product":
            product
    }