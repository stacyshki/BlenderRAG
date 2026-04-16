"""
DEMO PIPELINE
"""

import os
from dotenv import load_dotenv
from functools import partial
from src.load_embed_txt import load, get_model, emb_text
from src.load_DB import addInitDB
from src.prompt_engine import GroqLLM, RAGPipeline, launch_interface

# CONFIG
load_dotenv()
TEXT_FILE = "BlenderManual.txt"
DB_PATH = "./prod_db.db"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "intfloat/e5-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MIN_CHUNK_LEN = 80
BATCH_SIZE = 5000
N_RESULTS = 3
DIST_THRESHOLD = 0.8
LLM_MODEL = "openai/gpt-oss-120b"
GROQ_API_KEY = os.environ["GROQ_API_KEY"]


# RUN
docs = load(TEXT_FILE, "utf-8", CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LEN)

emb_model = get_model(EMBEDDING_MODEL)
emb_func = partial(emb_text, embedding_model=emb_model, norm=True)

collection = addInitDB(DB_PATH, COLLECTION_NAME, emb_func, docs, BATCH_SIZE)

llm = GroqLLM(api_key=GROQ_API_KEY, model=LLM_MODEL)

pipeline = RAGPipeline(
    collection=collection,
    embedding_func=emb_func,
    llm=llm,
    n_results=N_RESULTS,
    distance_threshold=DIST_THRESHOLD
)

launch_interface(pipeline, share=True)
