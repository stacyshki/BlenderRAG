"""
Evaluation pipeline for different RAG cfg comparisons
"""

import json
import os
from functools import partial
from src.load_embed_txt import load, get_model, emb_text
from src.load_DB import addInitDB
from src.prompt_engine import GroqLLM, T5LLM
from src.evaluate_results import (run_experiment, evaluate_locally,
            print_comparison_table, generate_configs, sample_configs)
from src.custom_embedding_ragas import CustomEmbeddings
from tqdm import tqdm
from dotenv import load_dotenv

# CONFIG
load_dotenv()

TEST_QUESTIONS = [
    {
        "question": "How do you switch between Object Mode and Edit Mode using the keyboard?",
        "ground_truth": "Press Tab to toggle between Object Mode and Edit Mode."
    },
    {
        "question": "What does pressing Ctrl+Tab do in Blender?",
        "ground_truth": "Ctrl+Tab opens the mode selection pie menu."
    },
    {
        "question": "Can you edit multiple objects at once in Edit Mode?",
        "ground_truth": "Yes, Blender supports multi-object editing in Edit Mode for compatible object types."
    },
    {
        "question": "What happens if you try to enter Edit Mode with a light selected?",
        "ground_truth": "Edit Mode is not available for lights, so nothing happens or Blender stays in Object Mode."
    },
    {
        "question": "What is Pose Mode used for?",
        "ground_truth": "Pose Mode is used to animate armatures by manipulating bones."
    },
    {
        "question": "Which object type supports Sculpt Mode?",
        "ground_truth": "Only mesh objects support Sculpt Mode."
    },
    {
        "question": "How do you select all elements in Edit Mode?",
        "ground_truth": "Press A to select all elements."
    },
    {
        "question": "How do you deselect everything in Edit Mode?",
        "ground_truth": "Press A again or Alt+A to deselect all elements."
    },
    {
        "question": "What is the purpose of Vertex Select, Edge Select, and Face Select modes?",
        "ground_truth": "They control whether you select vertices, edges, or faces in Edit Mode."
    },
    {
        "question": "How do you switch between Vertex, Edge, and Face selection modes?",
        "ground_truth": "Press 1, 2, or 3 on the keyboard (not numpad)."
    },
    {
        "question": "What does the G key do in Blender?",
        "ground_truth": "The G key activates the move (grab) tool."
    },
    {
        "question": "What does the R key do in Blender?",
        "ground_truth": "The R key activates rotation."
    },
    {
        "question": "What does the S key do in Blender?",
        "ground_truth": "The S key activates scaling."
    },
    {
        "question": "What is the difference between Global and Local transformation orientation?",
        "ground_truth": "Global uses world axes, while Local uses the object's own axes."
    },
    {
        "question": "What does pressing X or Delete do in Edit Mode?",
        "ground_truth": "It opens the delete menu to remove selected geometry."
    },
    {
        "question": "What is Extrude in Blender?",
        "ground_truth": "Extrude creates new geometry by extending selected elements."
    },
    {
        "question": "What shortcut is used to extrude geometry?",
        "ground_truth": "Press E to extrude selected elements."
    },
    {
        "question": "What does the Loop Cut tool do?",
        "ground_truth": "It adds a loop of edges across a mesh."
    },
    {
        "question": "What shortcut activates Loop Cut?",
        "ground_truth": "Press Ctrl+R to activate Loop Cut."
    },
    {
        "question": "What is proportional editing?",
        "ground_truth": "Proportional editing affects nearby vertices with a falloff when transforming."
    },
    {
        "question": "How do you enable proportional editing?",
        "ground_truth": "Press O to toggle proportional editing."
    },
    {
        "question": "What does Shade Smooth do?",
        "ground_truth": "It smooths the shading of an object without changing geometry."
    },
    {
        "question": "What is the purpose of the Subdivision Surface modifier?",
        "ground_truth": "It increases mesh resolution and smooths the surface."
    },
    {
        "question": "What is the difference between Object Mode and Sculpt Mode?",
        "ground_truth": "Object Mode transforms objects, while Sculpt Mode modifies mesh shape using brushes."
    },
    {
        "question": "What does the Tab key do when multiple modes are available?",
        "ground_truth": "It toggles between the last used modes, typically Object and Edit Mode."
    }
]

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
TEXT_FILE = "BlenderManual.txt"
SAVE_PATH = "results/eval_results.json"

EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
]

LLM_CONFIGS = {
    "t5": [
        "google/flan-t5-base",
        "google/flan-t5-large",
    ],
    "groq": [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
    ],
}

PARAM_GRID = {
    "chunk_size": [500, 1000, 1500, 2000],
    "chunk_overlap": [100, 200],
    "n_results": [3, 5, 10],
    "distance_threshold": [0.8, 1, 1.1],
    "content_len": [80, 100, 150]
}

# RUN
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "r") as f:
        all_results = json.load(f)
    done_configs = {r["name"] for r in all_results}
else:
    all_results = []
    done_configs = set()

configs = generate_configs(PARAM_GRID, EMBEDDING_MODELS, LLM_CONFIGS)
configs = sample_configs(configs, 25)
configs = [c for c in configs if c.name not in done_configs]

print(f"Skipping already computed configs: {len(done_configs)}")
print(f"Remaining configs to run: {len(configs)}")

for config in tqdm(configs, "Cfg processed"):
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"{'='*60}")
    
    docs = load(TEXT_FILE, "utf-8", config.chunk_size,
                config.chunk_overlap, config.content_len)
    
    emb_model = get_model(config.embedding_model)
    emb_func = partial(emb_text, embedding_model=emb_model, norm=True)
    embeddings_ragas = CustomEmbeddings(emb_model)
    
    collection_name = f"rag_{config.embedding_model.replace('/', '_').replace('-', '_')}"
    collection = addInitDB("./eval_db", collection_name, emb_func,
                            docs, batch_size=5000)
    
    if config.llm_backend == "groq":
        llm = GroqLLM(api_key=GROQ_API_KEY, model=config.llm_model)
    else:
        llm = T5LLM(model_name=config.llm_model)
    
    data, avg_dist = run_experiment(config, collection, emb_func,
                                    llm, TEST_QUESTIONS)
    
    scores = evaluate_locally(data, GROQ_API_KEY, embeddings_ragas)
    
    avg_ret = round(
        sum(data["retrieval_time"]) / len(data["retrieval_time"]),
        3
    )
    avg_gen = round(
        sum(data["generation_time"]) / len(data["generation_time"]),
        3
    )
    
    result_row = {
        "name": config.name,
        "avg_distance": avg_dist,
        "avg_retrieval_time": avg_ret,
        "avg_generation_time": avg_gen,
        **scores
    }
    all_results.append(result_row)
    
    with open(SAVE_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

print_comparison_table(all_results)

print(f"\nResults saved to {SAVE_PATH}")
