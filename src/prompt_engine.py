import gradio as gr
from abc import ABC, abstractmethod
from groq import Groq
from transformers import T5ForConditionalGeneration, T5Tokenizer


class BaseLLM(ABC):
    """Common interface for LLMs"""
    
    @abstractmethod
    def generate(self, context: str, question: str) -> str:
        pass


class GroqLLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant",
                max_tokens: int = 300):
        
        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
    
    def generate(self, context: str, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Blender documentation assistant. "
                        "Answer ONLY using the provided context. "
                        "If the context does not contain the answer, say: "
                        "'Not found in documentation.'"
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content


class T5LLM(BaseLLM):
    def __init__(self, model_name: str = "google/flan-t5-large",
                max_new_tokens: int = 300, max_input_length: int = 1024):
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length
    
    def generate(self, context: str, question: str) -> str:
        prompt = (
            f"Answer the question using the context below.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=self.max_input_length
        )
        outputs = self.model.generate(**inputs,
                                    max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


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


def launch_interface(pipeline: RAGPipeline, share: bool = False):
    gr.Interface(
        fn=pipeline.ask,
        inputs=gr.Textbox(label="Your question"),
        outputs=gr.Textbox(label="Answer"),
        title="Blender Documentation Assistant",
        flagging_mode="manual"
    ).launch(share=share)
