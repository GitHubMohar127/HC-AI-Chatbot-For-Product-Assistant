import re
import pandas as pd

from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "output"

DATASET_PATH = DATA_DIR / "hindcon_master_dataset.xlsx"


# ---------------------------------------------------------
# SECTION NAMES
# ---------------------------------------------------------

SECTION_NAMES = [
    "DESCRIPTION",
    "USES",
    "ADVANTAGES",
    "APPLICATION",
    "PROPERTIES",
    "COVERAGE",
    "SHELF LIFE",
    "PACKING",
    "HANDLING PRECAUTION",
    "HANDLING PRECAUTIONS",
    "STORAGE",
    "TECHNICAL DATA",
    "SPECIFICATIONS",
    "LIMITATIONS",
]


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_tds_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text
    )

    return text.strip()


# ---------------------------------------------------------
# HEADING NORMALIZATION
# ---------------------------------------------------------

def normalize_heading(text: str) -> str:

    text = text.strip().upper()

    text = re.sub(
        r"[^A-Z0-9 ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ---------------------------------------------------------
# CHECK SECTION HEADING
# ---------------------------------------------------------

def is_section_heading(line: str):

    normalized = normalize_heading(line)

    return normalized in SECTION_NAMES


# ---------------------------------------------------------
# EXTRACT SECTIONS
# ---------------------------------------------------------

def extract_sections(text: str):

    lines = text.split("\n")

    sections = {}

    current_section = "GENERAL"

    sections[current_section] = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        normalized = normalize_heading(line)

        if normalized in SECTION_NAMES:

            current_section = normalized

            if current_section not in sections:
                sections[current_section] = []

        else:

            sections[current_section].append(line)

    cleaned_sections = {}

    for section, content in sections.items():

        section_text = "\n".join(content).strip()

        if section_text:

            cleaned_sections[section] = section_text

    return cleaned_sections


# ---------------------------------------------------------
# LOAD EXCEL DATA
# ---------------------------------------------------------

def load_valid_tds():

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"TDS dataset not found at: {DATASET_PATH}"
        )

    df = pd.read_excel(
        DATASET_PATH
    )

    valid_tds = df[
        df["TDS_Text"].notna()
        &
        (
            df["TDS_Text"]
            .astype(str)
            .str.strip()
            != ""
        )
        &
        (
            df["TDS_Status"]
            .astype(str)
            .str.lower()
            == "success"
        )
    ].copy()

    return valid_tds


# ---------------------------------------------------------
# CREATE DOCUMENTS
# ---------------------------------------------------------

def create_documents():

    valid_tds = load_valid_tds()

    documents = []

    for _, row in valid_tds.iterrows():

        product_id = str(
            row["Product_ID"]
        ).strip()

        product_name = str(
            row["Product_Name"]
        ).strip()

        category = str(
            row["Category"]
        ).strip()

        tds_link = str(
            row["TDS_Link"]
        ).strip()

        raw_text = clean_tds_text(
            row["TDS_Text"]
        )

        sections = extract_sections(
            raw_text
        )

        for section, section_text in sections.items():

            document = Document(

                page_content=section_text,

                metadata={

                    "product_id":
                        product_id,

                    "product_name":
                        product_name,

                    "category":
                        category,

                    "tds_link":
                        tds_link,

                    "tds_status":
                        str(row["TDS_Status"]),

                    "document_name":
                        f"{product_id}_TDS",

                    "section":
                        section,
                }
            )

            documents.append(
                document
            )

    return documents


# ---------------------------------------------------------
# CHUNK DOCUMENTS
# ---------------------------------------------------------

def create_chunks():

    documents = create_documents()

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1200,

        chunk_overlap=150,

        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " "
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    for i, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = i

    return chunks


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    chunks = create_chunks()

    print(
        f"Total chunks created: {len(chunks)}"
    )

    if chunks:

        print(
            "\nFirst chunk:"
        )

        print(
            chunks[0].page_content
        )

        print(
            "\nMetadata:"
        )

        print(
            chunks[0].metadata
        )



# if __name__ == "__main__":
#     chunks = create_chunks()
#     print(f"Total chunks created: {len(chunks)}")