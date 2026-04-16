import time
from dataclasses import dataclass, field
from numpy import average, mean
from itertools import product
import random
import re

random.seed(42)


@dataclass
class EvalConfig:
    """Evaluation experiment"""
    name: str
    embedding_model: str
    llm_backend: str # "groq" or "t5" that I did for this project
    llm_model: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    n_results: int = 5
    distance_threshold: float = 0.8
    results: list = field(default_factory=list)
    content_len: int = 100


def run_experiment(config: EvalConfig, collection,
                    embedding_func, llm,
                    test_questions) -> tuple[dict, float]:
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "retrieval_time": [],
        "generation_time": [],
    }
    distance = []
    
    for item in test_questions:
        # RETRIEVAL
        question = item["question"]
        
        t0 = time.time()
        query_emb = embedding_func(question)
        results = collection.query(query_embeddings=[query_emb],
                                    n_results=config.n_results)
        retrieval_time = time.time() - t0
        
        chunks = results['documents'][0]
        context = "\n".join(chunks)
        
        # GENERATION
        t0 = time.time()
        answer = llm.generate(context, question)
        generation_time = time.time() - t0
        
        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(chunks)
        data["ground_truth"].append(item["ground_truth"])
        data["retrieval_time"].append(round(retrieval_time, 3))
        data["generation_time"].append(round(generation_time, 3))
        
        print(f"  Q: {question[:60]}...")
        print(f"  A: {answer[:100]}...")
        print(f"  Retrieval: {retrieval_time:.2f}s | Generation: {generation_time:.2f}s\n")
        
        distance.append(results['distances'][0][0])
    
    return data, float(average(distance)) # I check avg min distance


def cosine(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot / (norm1 * norm2)


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    
    if not a and not b:
        return 1.0
    
    union = a | b
    
    if not union:
        return 0.0
    
    return len(a & b) / len(union)


def evaluate_locally(data: dict, embeddings) -> dict:
    """
    Eval on embedding-based proxy metrics
    """
    faithfulness_scores = []
    answer_relevancy_scores = []
    context_precision_scores = []
    
    for question, answer, contexts, ground_truth in zip(
        data["question"],
        data["answer"],
        data["contexts"],
        data["ground_truth"],
    ):
        q_emb = embeddings.embed_query(question)
        a_emb = embeddings.embed_query(answer)
        gt_emb = embeddings.embed_query(ground_truth)
        ctx_embs = embeddings.embed_documents(contexts) if contexts else []
        
        # Faithfulness proxy
        if ctx_embs:
            faith = max(cosine(a_emb, c_emb) for c_emb in ctx_embs)
        else:
            faith = 0.0
        
        # Answer relevancy proxy
        relevancy = cosine(a_emb, q_emb)
        
        # Context precision proxy: fraction of retrieved chunks close to GT
        if ctx_embs:
            gt_sim = [cosine(gt_emb, c_emb) for c_emb in ctx_embs]
            sem_prec = sum(sim >= 0.55 for sim in gt_sim) / len(gt_sim)
        else:
            sem_prec = 0.0
        
        # Lexical signal so very short answers/contexts are not overrated
        gt_tokens = token_set(ground_truth)
        ctx_tokens = token_set(" ".join(contexts))
        lex_prec = jaccard(gt_tokens, ctx_tokens)
        precision = 0.7 * sem_prec + 0.3 * lex_prec
        
        faithfulness_scores.append(max(0.0, min(1.0, faith)))
        answer_relevancy_scores.append(max(0.0, min(1.0, relevancy)))
        context_precision_scores.append(max(0.0, min(1.0, precision)))
    
    return {
        "faithfulness": round(float(mean(faithfulness_scores)), 3),
        "answer_relevancy": round(float(mean(answer_relevancy_scores)), 3),
        "context_precision": round(float(mean(context_precision_scores)), 3),
    }


def print_comparison_table(all_results: list[dict]):
    print("\n" + "=" * 80)
    print(f"{'Config':<25} {'Faithful':>10} {'Relevancy':>10} {'Precision':>10} {'Avg Ret(s)':>12} {'Avg Gen(s)':>12}")
    print("=" * 80)
    
    for r in all_results:
        print(
            f"{r['name']:<25} "
            f"{r.get('faithfulness', 'N/A'):>10} "
            f"{r.get('answer_relevancy', 'N/A'):>10} "
            f"{r.get('context_precision', 'N/A'):>10} "
            f"{r.get('avg_retrieval_time', 'N/A'):>12} "
            f"{r.get('avg_generation_time', 'N/A'):>12}"
        )
    
    print("=" * 80)


def generate_configs(param_grid: dict, embedding_models: list,
                    llm_configs: dict) -> list:
    configs = []
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for emb_model in embedding_models:
        for backend, llm_models in llm_configs.items():
            for llm_model in llm_models:
                for combo in product(*param_values):
                    params = dict(zip(param_names, combo))
                    
                    name = (
                        f"{emb_model.split('/')[-1]} + "
                        f"{llm_model.split('/')[-1]} | "
                        f"cs={params['chunk_size']} "
                        f"ov={params['chunk_overlap']} "
                        f"n={params['n_results']} "
                        f"th={params['distance_threshold']}"
                    )
                    
                    configs.append(
                        EvalConfig(
                            name=name,
                            embedding_model=emb_model,
                            llm_backend=backend,
                            llm_model=llm_model,
                            **params
                        )
                    )
    
    return configs


def sample_configs(configs, k=20) -> list:
    return random.sample(configs, k)
