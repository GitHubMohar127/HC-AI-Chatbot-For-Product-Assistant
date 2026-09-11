import re
import pandas as pd

from pathlib import Path
from rapidfuzz import fuzz


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "output"
    / "hindcon_master_dataset.xlsx"
)


# ============================================================
# 2. LOAD PRODUCT MASTER
# ============================================================

df = pd.read_excel(DATASET_PATH)


valid_tds = df[
    df["TDS_Text"].notna()
    & (
        df["TDS_Text"]
        .astype(str)
        .str.strip()
        != ""
    )
    & (
        df["TDS_Status"]
        .astype(str)
        .str.lower()
        == "success"
    )
].copy()


product_master = (
    valid_tds[
        [
            "Product_ID",
            "Product_Name",
            "Category"
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)


# ============================================================
# 3. TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")

    text = re.sub(
        r"[^a-z0-9\s-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 4. PRODUCT QUERY NORMALIZATION
# ============================================================

def normalize_product_query(text: str) -> str:

    text = normalize_text(text)

    replacements = {
        "crystal": "crystel"
    }

    words = text.split()

    corrected_words = [
        replacements.get(
            word,
            word
        )
        for word in words
    ]

    return " ".join(
        corrected_words
    )


# ============================================================
# 5. PRODUCT LOOKUP
# ============================================================

product_lookup = {}


for _, row in product_master.iterrows():

    product_id = str(
        row["Product_ID"]
    ).strip()

    product_name = str(
        row["Product_Name"]
    ).strip()

    normalized_name = (
        normalize_product_query(
            product_name
        )
    )

    product_lookup[
        normalized_name
    ] = {

        "product_id":
            product_id,

        "product_name":
            product_name,

        "category":
            str(
                row["Category"]
            ).strip()
    }


# ============================================================
# 6. PRODUCT DETECTION
# ============================================================

def detect_product(query: str):
    """
    Detect a Hindcon product from the user query.

    Matching order:
    1. Exact product-name match
    2. Fuzzy product-name match for spelling mistakes
    3. Ambiguous match detection

    Returns:
        FOUND
        AMBIGUOUS
        NOT_FOUND
    """

    normalized_query = normalize_product_query(query)

    # ---------------------------------------------------------
    # STEP 1: EXACT MATCH
    # ---------------------------------------------------------

    exact_matches = []

    for normalized_name, product_info in product_lookup.items():

        if normalized_name in normalized_query:
            exact_matches.append(
                (len(normalized_name), product_info)
            )

    if exact_matches:

        # Prefer the longest product name.
        # This is important for products such as:
        #
        # Hind Crystel Seal
        # Hind Crystel Seal (1)
        #
        exact_matches.sort(
            key=lambda x: x[0],
            reverse=True
        )

        longest_length = exact_matches[0][0]

        best_matches = [
            product_info
            for length, product_info in exact_matches
            if length == longest_length
        ]

        # Remove duplicate Product IDs
        unique_products = {}

        for product_info in best_matches:
            unique_products[
                product_info["product_id"]
            ] = product_info

        best_matches = list(
            unique_products.values()
        )

        if len(best_matches) == 1:

            return {
                "status": "FOUND",
                "product": best_matches[0],
                "match_type": "EXACT",
                "confidence": 100
            }

        return {
            "status": "AMBIGUOUS",
            "products": best_matches,
            "match_type": "EXACT",
            "confidence": 100
        }

    # ---------------------------------------------------------
    # STEP 2: FUZZY MATCH
    # ---------------------------------------------------------

    fuzzy_matches = []

    query_words = set(
        normalized_query.split()
    )

    for normalized_name, product_info in product_lookup.items():

        product_words = set(
            normalized_name.split()
        )

        # -----------------------------------------------------
        # Token-based similarity
        # -----------------------------------------------------
        #
        # Example:
        #
        # hind mould release form tr3
        #
        # hind mould ralease form tr3
        #
        # These are highly similar even though "release"
        # is misspelled.
        #

        token_score = fuzz.token_set_ratio(
            normalized_name,
            normalized_query
        )

        # -----------------------------------------------------
        # Partial similarity
        # -----------------------------------------------------

        partial_score = fuzz.partial_ratio(
            normalized_name,
            normalized_query
        )

        # -----------------------------------------------------
        # Choose the stronger score
        # -----------------------------------------------------

        similarity_score = max(
            token_score,
            partial_score
        )

        # -----------------------------------------------------
        # Product-name word overlap
        # -----------------------------------------------------

        common_words = (
            query_words & product_words
        )

        word_overlap = len(common_words)

        # -----------------------------------------------------
        # Require reasonable similarity
        # -----------------------------------------------------

        if similarity_score >= 80 and word_overlap >= 2:

            fuzzy_matches.append(
                (
                    similarity_score,
                    word_overlap,
                    product_info
                )
            )

    # ---------------------------------------------------------
    # STEP 3: NO FUZZY MATCH
    # ---------------------------------------------------------

    if not fuzzy_matches:

        return {
            "status": "NOT_FOUND",
            "products": [],
            "match_type": "NONE",
            "confidence": 0
        }

    # ---------------------------------------------------------
    # STEP 4: SORT FUZZY MATCHES
    # ---------------------------------------------------------

    fuzzy_matches.sort(
        key=lambda x: (
            x[0],
            x[1]
        ),
        reverse=True
    )

    best_score = fuzzy_matches[0][0]

    # Keep products very close to the best score.
    close_matches = [
        item
        for item in fuzzy_matches
        if item[0] >= best_score - 3
    ]

    # Remove duplicate Product IDs

    unique_products = {}

    for score, overlap, product_info in close_matches:

        product_id = product_info["product_id"]

        if product_id not in unique_products:

            unique_products[product_id] = {
                "product": product_info,
                "score": score,
                "overlap": overlap
            }

    close_matches = list(
        unique_products.values()
    )

    # ---------------------------------------------------------
    # STEP 5: HIGH-CONFIDENCE SINGLE MATCH
    # ---------------------------------------------------------

    if len(close_matches) == 1:

        best = close_matches[0]

        return {
            "status": "FOUND",
            "product": best["product"],
            "match_type": "FUZZY",
            "confidence": best["score"]
        }

    # ---------------------------------------------------------
    # STEP 6: AMBIGUOUS FUZZY MATCH
    # ---------------------------------------------------------

    return {
        "status": "AMBIGUOUS",
        "products": [
            item["product"]
            for item in close_matches
        ],
        "match_type": "FUZZY",
        "confidence": best_score
    }


# ============================================================
# 7. KNOWN PRODUCT WORDS
# ============================================================

known_product_words = set()


for _, row in product_master.iterrows():

    product_name = (
        normalize_product_query(
            str(
                row["Product_Name"]
            )
        )
    )


    for word in product_name.split():

        if len(word) >= 3:

            known_product_words.add(
                word
            )


# ============================================================
# 8. PRODUCT-LIKE QUERY DETECTION
# ============================================================

def query_contains_product_like_phrase(
    query: str
):

    normalized_query = (
        normalize_product_query(
            query
        )
    )

    words = normalized_query.split()


    for word in words:

        if word in known_product_words:

            return True


    return False


# ============================================================
# 9. PRODUCT STATUS
# ============================================================

def get_product_status(query: str):

    product_result = detect_product(
        query
    )


    if (
        product_result["status"]
        == "FOUND"
    ):

        return {

            "status": "FOUND",

            "product":
                product_result["product"]
        }


    if (
        product_result["status"]
        == "AMBIGUOUS"
    ):

        return {

            "status": "AMBIGUOUS",

            "products":
                product_result["products"]
        }


    if query_contains_product_like_phrase(
        query
    ):

        return {

            "status":
                "PRODUCT_NOT_FOUND",

            "products": []
        }


    return {

        "status":
            "NOT_SPECIFIED",

        "products": []
    }


# ============================================================
# 10. SECTION ALIASES
# ============================================================

SECTION_ALIASES = {

    "COVERAGE": [

        "coverage",
        "consumption",
        "quantity required",
        "material required",
        "how much material",
        "application rate"
    ],

    "SHELF LIFE": [

        "shelf life",
        "storage life",
        "expiry",
        "expiration",
        "validity"
    ],

    "PACKING": [

        "packing",
        "package",
        "pack size",
        "packaging"
    ],

    "DESCRIPTION": [

        "description",
        "what does it do",
        "describe the product"
    ],

    "USES": [

        "uses",
        "used for",
        "where is it used",
        "purpose"
    ],

    "ADVANTAGES": [

        "advantages",
        "benefits",
        "features"
    ],

    "APPLICATION": [

        "application",
        "how to apply",
        "application method",
        "procedure",
        "method of application"
    ],

    "PROPERTIES": [

        "properties",
        "physical properties",
        "technical properties"
    ],

    "TECHNICAL DATA": [

        "technical data",
        "technical information"
    ],

    "SPECIFICATIONS": [

        "specifications",
        "specification"
    ],

    "LIMITATIONS": [

        "limitations",
        "limitations of use",
        "restrictions"
    ]
}


# ============================================================
# 11. SECTION DETECTION
# ============================================================

def detect_section(query: str):

    normalized_query = normalize_text(
        query
    )

    matches = []


    for section, aliases in (
        SECTION_ALIASES.items()
    ):

        for alias in aliases:

            if normalize_text(alias) in normalized_query:

                matches.append(
                    section
                )

                break


    matches = list(
        dict.fromkeys(matches)
    )


    if len(matches) == 0:

        return {
            "status": "NOT_FOUND",
            "section": None,
            "sections": []
        }


    if len(matches) == 1:

        return {
            "status": "FOUND",
            "section": matches[0],
            "sections": matches
        }


    return {
        "status": "MULTIPLE",
        "section": None,
        "sections": matches
    }


# ============================================================
# 12. QUERY STATUS
# ============================================================

QUERY_STATUS = {

    "VALID":
        "VALID",

    "INCOMPLETE_QUERY":
        "INCOMPLETE_QUERY",

    "PRODUCT_NOT_FOUND":
        "PRODUCT_NOT_FOUND",

    "INFORMATION_NOT_AVAILABLE":
        "INFORMATION_NOT_AVAILABLE",

    "INVALID_QUERY":
        "INVALID_QUERY"
}


# ============================================================
# 13. RESPONSE MESSAGES
# ============================================================

RESPONSE_MESSAGES = {

    "INCOMPLETE_QUERY":
        "Please specify which product you mean.",

    "PRODUCT_NOT_FOUND":
        "The specified product was not found in the Hindcon TDS database.",

    "INFORMATION_NOT_AVAILABLE":
        "The requested information is not available in the provided TDS.",

    "INVALID_QUERY":
        "Invalid question. Please ask a question related to a Hindcon product or its TDS.",

    "AMBIGUOUS_PRODUCT":
        "Multiple products match your question. Please specify the exact product name."
}


# ============================================================
# 14. BASIC QUERY VALIDATION
# ============================================================

def is_obviously_invalid_query(
    query: str
):

    if not isinstance(
        query,
        str
    ):

        return True


    query = query.strip()


    if not query:

        return True


    if len(query) < 3:

        return True


    return False


# ============================================================
# 15. CONVERSATION PRODUCT CONTEXT
# ============================================================

def get_product_with_context(
    query: str,
    state: dict
):

    product_status = (
        get_product_status(
            query
        )
    )


    # Product explicitly mentioned
    if (
        product_status["status"]
        == "FOUND"
    ):

        return {

            "status": "FOUND",

            "product":
                product_status["product"],

            "source":
                "CURRENT_QUERY"
        }


    # Multiple products
    if (
        product_status["status"]
        == "AMBIGUOUS"
    ):

        return {

            "status":
                "AMBIGUOUS",

            "products":
                product_status["products"],

            "source":
                "CURRENT_QUERY"
        }


    # Product looks like a product
    # but was not found
    if (
        product_status["status"]
        == "PRODUCT_NOT_FOUND"
    ):

        return {

            "status":
                "PRODUCT_NOT_FOUND",

            "products": [],

            "source":
                "CURRENT_QUERY"
        }


    # Use previous conversation product
    if state.get("product_id"):

        return {

            "status": "FOUND",

            "product": {

                "product_id":
                    state["product_id"],

                "product_name":
                    state["product_name"],

                "category":
                    state["category"]
            },

            "source":
                "CONVERSATION_CONTEXT"
        }


    # No product anywhere
    return {

        "status":
            "NOT_SPECIFIED",

        "products": [],

        "source":
            "NONE"
    }


# ============================================================
# 16. UPDATE CONVERSATION STATE
# ============================================================

def update_conversation_state(
    state: dict,
    product: dict
):

    state["product"] = product

    state["product_id"] = (
        product["product_id"]
    )

    state["product_name"] = (
        product["product_name"]
    )

    state["category"] = (
        product["category"]
    )

    return state


# ============================================================
# 17. CONTEXT-AWARE ROUTER
# ============================================================

def route_query_with_context(
    query: str,
    state: dict
):

    # Basic validation
    if is_obviously_invalid_query(
        query
    ):

        return {

            "status":
                QUERY_STATUS[
                    "INVALID_QUERY"
                ],

            "message":
                RESPONSE_MESSAGES[
                    "INVALID_QUERY"
                ]
        }


    # Find product
    product_result = (
        get_product_with_context(
            query,
            state
        )
    )


    # Product not found
    if (
        product_result["status"]
        == "PRODUCT_NOT_FOUND"
    ):

        return {

            "status":
                QUERY_STATUS[
                    "PRODUCT_NOT_FOUND"
                ],

            "message":
                RESPONSE_MESSAGES[
                    "PRODUCT_NOT_FOUND"
                ]
        }


    # Ambiguous product
    if (
        product_result["status"]
        == "AMBIGUOUS"
    ):

        return {

            "status":
                QUERY_STATUS[
                    "INCOMPLETE_QUERY"
                ],

            "message":
                RESPONSE_MESSAGES[
                    "AMBIGUOUS_PRODUCT"
                ]
        }


    # Product missing
    if (
        product_result["status"]
        == "NOT_SPECIFIED"
    ):

        return {

            "status":
                QUERY_STATUS[
                    "INCOMPLETE_QUERY"
                ],

            "message":
                RESPONSE_MESSAGES[
                    "INCOMPLETE_QUERY"
                ]
        }


    # Valid product
    product = (
        product_result["product"]
    )


    # Update conversation memory
    update_conversation_state(
        state,
        product
    )


    # Detect section
    section_result = detect_section(
        query
    )


    return {

        "status":
            QUERY_STATUS["VALID"],

        "product":
            product,

        "section_detection":
            section_result,

        "product_source":
            product_result["source"]
    }


# ============================================================
# 18. BUILD CONTEXTUAL QUERY
# ============================================================

def build_contextual_query(
    query: str,
    product: dict
):

    product_name = (
        product["product_name"]
    )

    normalized_query = normalize_text(
        query
    )


    # Use complete-word matching,
    # not simple substring matching.
    tokens = set(
        normalized_query.split()
    )


    reference_words = {

        "it",
        "its",
        "this",
        "that"
    }


    contains_reference = (
        len(
            tokens.intersection(
                reference_words
            )
        ) > 0
    )


    if contains_reference:

        return (
            f"{query} "
            f"for {product_name}"
        )


    return query



# result = detect_product(
#     "tell me about Hind Mould Ralease Form TR3 this product"
# )

# print(result)