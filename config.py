import os

from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone


load_dotenv()


# ==============================
# GROQ
# ==============================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")


groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ==============================
# PINECONE
# ==============================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME"
)


if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env")


if not PINECONE_INDEX_NAME:
    raise ValueError(
        "PINECONE_INDEX_NAME not found in .env"
    )


pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY
)