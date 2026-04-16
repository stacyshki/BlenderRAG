from langchain.embeddings.base import Embeddings

class CustomEmbeddings(Embeddings):
    def __init__(self, emb_model):
        self.emb_model = emb_model
    
    def embed_query(self, text: str):
        return self.emb_model.encode([text],
                                    normalize_embeddings=True)[0].tolist()
    
    def embed_documents(self, texts: list[str]):
        return self.emb_model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()
