# Blender RAG project

**Link to demo:** [link](https://huggingface.co/spaces/stacyshki/Blender-RAG)

The project is designed to use retrieval-augmented generation for the purpose of extracting information from Blender. Blender is a free and open-source 3D computer graphics software suite. It runs on Windows, macOS, Linux, BSD, Haiku, and IRIX and is used for general 3D modeling, animation, digital sculpting, visual effects, rendering, compositing, motion tracking, drawing, 2D animation and video editing. It is widely used to create animated films, visual effects, art, 3D-printed models, motion graphics, interactive 3D applications, virtual reality, and 3D models for video games.

The manual in use can be found [here](https://docs.blender.org/manual/en/latest/index.html), and it mostly navigates users on how to use the application and what functionality is available there.

## Contents

- [Purpose of the project](#purpose-of-the-project)
- [Deployment](#deployment)
- [Limitations and solutions](#limitations-and-solutions)
  - [Designed approaches](#designed-approaches)
- [Main configuration](#main-configuration)
  - [Comments on the configuration](#comments-on-the-configuration)
- [Repository structure](#repository-structure)
- [Cloning/Forking setup](#cloningforking-setup)
- [Important notes](#important-notes)

## Purpose of the project

**Blender RAG** is a question-answering system built on top of the Blender 5.1 Reference Manual. It combines semantic search with large language models to provide accurate, documentation-grounded answers to questions about Blender's features, tools, and workflows.

The system uses ChromaDB to index and retrieve relevant chunks from the manual using `intfloat/e5-base-v2` embeddings, and passes the top-3 retrieved results as context to `GPT-OSS 120B` which generates a concise answer. A distance threshold filter ensures the model only responds to Blender-related questions and does not rely on its own training knowledge.

The pipeline covers the full RAG workflow: document loading and chunking, embedding generation, vector storage and retrieval, prompt engineering, and response generation with a Gradio interface for user interaction. The final configuration was selected based on evaluation across multiple embedding models, chunk sizes, and retrieval parameters using custom evaluation metrics.

## Deployment

This is the main repository for the code base.

However, there exists a deployment of the app on Hugging Face, which you can find visiting [this link](https://huggingface.co/spaces/stacyshki/Blender-RAG).

The main branch on Github slightly differs from the one on Hugging Face in commits and file tree. On Hugging Face you can additionally find stored production database (provided via git lfs) with supporting files, and the README is different, designed to start HF app.

[Back to contents](#contents)

## Limitations and solutions

#### **Chunking problem:**

There are a lot of lines on the manual which are either too short or contain empty strings `"\n"`, generating weak chunks.

**Solution:**

Filter chunks and try to use bigger chunk_sizes.

---

#### **Answer generation models:**

Modern LLMs are too have to run on-premise with personal computer. Meanwhile, weaker LLMs give either too shallow answers or cannot fully interpret the provided context.

**Solution:**

In-production pipeline uses Groq free-tier API as an inference provider. Also, there is a possibility to use Hugging Face for inference, but these attempts are not included in the final code base of this repository.

**Limitation:**

It is easy to run out of free tokens if demand is high.

---

#### **Retrieved data:**

Some questions cannot be answered with the top-n similarities from the database. For example, these could be particular questions about shortcuts. The reason is that a shortcut itself is usually emphasized with linebreaks around it (e.g. "\n\nToggle Edit Mode: Tab\n\n"), taking a separate chunk or a lot of useless place in a chunk. At the same time, there could be a chunk with a plenty of information about the same topic except the shortcut itself (for example, a long paragraph about Edit Mode). This forces retrieval algorithm to give back the paragraph, not the shortcut, as it is contextually 'closer' to the question.

**Solution:**

Experiment with `n_results` from this code:

```python
results = self.collection.query(
          query_embeddings=[query_embedding],
          n_results=self.n_results
      )
```

\- forcing the query to give back more results, even if they could seem less helpful for the question.

**Limitation:**

If we use more chunks, then we consume much more tokens, while in most cases adding no real value. If we use less chunks, we could lose potentially important information.

---

#### **Prompt problems:**

As LLMs are trained to be too helpful and to answer fluently, it is impossible to restrict the model from answering "What is a bubble sort algorithm", etc. - questions, irrelevant to the RAG topic. Moreover, we would want to limit model's answering capabilities if a question cannot be answered with the existing documentation, preventing the model from overusing its training knowledge and preventing it from particular hallucinations when a model makes facts up due to insufficient knowledge base.

**Solution:**

Firstly, I still tried to limit models via setting roles or experimenting with prompt structures. However, it appeared to be too easy to expose this and surpass any in-prompt restrictions.

Secondly, I decided to limit this problem before it reaches the model via this code:

```python
best_distance = results['distances'][0][0]
if best_distance > self.distance_threshold:
    return "This question doesn't seem to be about Blender."
```

More on that could be found in [designed approaches](#designed-approaches).

**Limitation:**

The threshold is sensitive to the embeddings and distances, and in some cases, if set too strictly, it prevents model from answering real questions on Blender that are mentioned in the manual, but at the same time have little description in the docs or answer the question indirectly.

---

#### **Evaluation:**

The next problem was to understand how well does a model perform, as it can hallucinate or simply the distance threshold discussed above may be too strict.

**Solution:**

Use an evaluation pipeline before production based on 25 questions with ground truth provided. Initially, it was supposed to use RAGAS for evaluation, using a Groq API to have a model for evaluation. It turned out that I was running out of tokens too fast, creating an extra difficulty to complete a step of evaluation. As a result, I implemented a local alternative to RAGAS, avoiding the need for an external LLM-as-a-judge. The evaluation is fully based on embeddings and lightweight lexical metrics, making it fast and cost-efficient.

1. Faithfulness: measures how grounded the answer is in the retrieved context. The embedding of the generated answer is compared against embeddings of all retrieved chunks using cosine similarity. The maximum similarity score is taken, assuming that if the answer is close to at least one chunk, it is likely derived from the context.
2. Answer Relevancy: evaluates how relevant the answer is to the original question. This is computed as the cosine similarity between the question embedding and the answer embedding, based on the assumption that a good answer should be semantically aligned with the question.
3. Context Precision: measures how well the retrieval step identifies useful context. It is computed as a weighted combination of:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;semantic signal (70%): proportion of chunks with cosine similarity ≥ 0.55 to the ground truth

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;lexical signal (30%): Jaccard similarity between tokens of the ground truth and retrieved chunks

**Limitations:**

While this approach is efficient and fully local, it is less accurate than LLM-based evaluation. Faithfulness in particular is weak, as taking the maximum similarity across chunks is a soft criterion and may fail to detect hallucinations when embeddings are still close in vector space.

[Back to contents](#contents)

### Designed approaches

Evaluation pipeline tested 25 different sampled configs with the following parameters:

```python
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
```

- EMBEDDING_MODELS: list of models from Hugging Face, used in embedding/retrieval.
- LLM_CONFIGS: dictionary with T5 LLMs from Hugging Face and LLMs from Groq API. A separate module was coded to differentiate these two platforms, ensuring correct prompt-generation functionality is used to utilize them.
- PARAM_GRID: dictionary with other parameters used in DB creation/answer generation system
  - chunk_size and chunk_overlap: parameters used in splitting documents

    ```python
    splitter = RecursiveCharacterTextSplitter(
          chunk_size=chunk_size,
          chunk_overlap=overlap
      )
    ```

  - content_len: parameter used for reduction of chunks to filter out titles and multiple linebreaks
    ```python
    prepFile = [c.page_content for c in split_docs
                  if len(c.page_content.strip()) > min_content_len]
    ```
  - n_results: parameter used for retrieval, indicating the amount of similar results to be drawn from DB
    ```python
    results = self.collection.query(
              query_embeddings=[query_embedding],
              n_results=self.n_results
          )
    ```
  - distance_threshold: parameter used to limit LLM generation capabilities, and restrict its answers only to downloaded Blender manual. The code forces a return in case of weak match
    ```python
    best_distance = results['distances'][0][0]
    if best_distance > self.distance_threshold:
        return "This question doesn't seem to be about Blender."
    ```

**In the root you will also find analysis of the evaluation (`evaluation_analysis.ipynb`), which aimed to figure out the best config for production.**

Production pipeline (main.py) utilizes best configuration (see [cfg](#main-configuration)) to execute custom RAG pipeline with Gradio:

```python
class RAGPipeline:
    def __init__(self, collection, embedding_func, llm: BaseLLM,
                n_results: int = 5, distance_threshold: float = 0.95):

        self.collection = collection
        self.embedding_func = embedding_func
        self.llm = llm
        self.n_results = n_results
        self.distance_threshold = distance_threshold

    def ask(self, question: str) -> str:
        query_embedding = self.embedding_func(question)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.n_results
        )

        best_distance = results['distances'][0][0]
        if best_distance > self.distance_threshold:
            return "This question doesn't seem to be about Blender."

        context = "\n".join(results['documents'][0])
        return self.llm.generate(context, question)


# GRADIO


def launch_interface(pipeline: RAGPipeline, share: bool = False,
                    server_name: str = "127.0.0.1", port: int = 7860):
    gr.Interface(
        fn=pipeline.ask,
        inputs=gr.Textbox(label="Your question"),
        outputs=gr.Textbox(label="Answer"),
        title="Blender Documentation Assistant",
        flagging_mode="manual"
    ).launch(share=share, server_name=server_name, server_port=port)
```

**Note: for Hugging Face deployment it is not necessary to configure `.launch()` parameters (share, server_name, server_port). Initially, I intended to use Render PaaS for deployment, which requires these settings.**

[Back to contents](#contents)

## Main configuration

| Configuration setting | Value               | Comments                                                          |
| --------------------- | ------------------- | ----------------------------------------------------------------- |
| TEXT_FILE             | BlenderManual.txt   | Source for my database _(1)_                                      |
| EMBEDDING_MODEL       | intfloat/e5-base-v2 | Setting from best cfg _(2)_                                       |
| CHUNK_SIZE            | 1000                | Setting from best cfg _(2)_                                       |
| CHUNK_OVERLAP         | 100                 | Setting from best cfg _(2)_                                       |
| MIN_CHUNK_LEN         | 80                  | Empirical setting based on specifics of the documentation _(3)_   |
| BATCH_SIZE            | 5000                | Batching for DB upload                                            |
| N_RESULTS             | 3                   | Setting from best cfg _(2)_                                       |
| DIST_THRESHOLD        | 0.8                 | Threshold used to limit text generation model functionality _(4)_ |
| LLM_MODEL             | openai/gpt-oss-120b | Text generation model _(5)_                                       |

### _Comments on the configuration_

_(1)_ The file was obtained from [Blender.org](https://docs.blender.org/manual/en/latest/index.html) as EPUB and then converted to TXT with [Calibre](https://calibre-ebook.com/), and it serves as the source for the database

_(2)_ Best config is decided in `evaluation_analysis.ipynb` notebook

_(3)_ Parameter addressing issue described [here](#chunking-problem)

_(4)_ Parameter addressing issue described [here](#prompt-problems)

_(5)_ Model using Groq API interface

[Back to contents](#contents)

## Repository structure

```text
BlenderRAG/
├── results/                         # Evaluation pipeline outcomes
│   └── eval_results.json
├── src/                             # Main source code
│   ├── custom_embedding_ragas.py    # Embedding creation for local evaluation
│   ├── evaluate_results.py          # Logic for evaluation
│   ├── load_DB.py                   # Database initialization
│   ├── load_embed_txt.py            # Loading and embedding for BlenderManual.txt
│   └── prompt_engine.py             # Main answer generation logic
├── .env.example                     # Example how the .env file should look like
├── .gitignore
├── BlenderManual.txt                # Source text for database
├── README.md
├── evaluation.py                    # Evaluation pipeline writing to results/eval_results.json
├── evaluation_analysis.ipynb        # Analysis of results/eval_results.json
├── main.py                          # Production pipeline using Gradio
└── requirements.txt                 # Python dependencies
```

## Cloning/Forking setup

### Step 1

#### Option 1: Clone the repository

```bash
git clone https://github.com/stacyshki/BlenderRAG.git
cd BlenderRAG
```

#### Option 2: Fork the repository

1. Open the repository on GitHub and click Fork.
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/BlenderRAG.git
   cd BlenderRAG
   ```

### Step 2: Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Step 3: Configure environment

```bash
cp .env.example .env
```

Then edit `.env`:

```text
GROQ_API_KEY=your_groq_api_key_here
```

### Step 4: Run

```bash
python main.py
```

### Notes

- The app uses BlenderManual.txt as the source document
- The local database is stored in prod_db.db
- By default, the interface starts on port 7860
- If you do not want to use Groq API, you can use T5 Hugging Face model:

  ```python
  # in main.py:
  from src.prompt_engine import T5LLM
  ...
  llm =T5LLM()

  pipeline = RAGPipeline(
      collection=collection,
      embedding_func=emb_func,
      llm=llm,
      n_results=N_RESULTS,
      distance_threshold=DIST_THRESHOLD
  )
  ```

## Important notes

- Database can be committed only with git lfs
- Hugging Face token is not set up here. Nevertheless, it can be implemented to make embedding model/T5 model load faster (it is still instant without the token for me)
- Not [all possible configurations](#designed-approaches) are tested, they are sampled with `seed=42`

[Back to contents](#contents)
