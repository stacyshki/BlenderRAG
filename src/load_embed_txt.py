from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer


def load(text: str, encoding: str,
            chunk_size: int, overlap: int,
            min_content_len: int
        ) -> list:
    loader = TextLoader(text, encoding=encoding)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    split_docs = splitter.split_documents(docs)
    
    prepFile = [c.page_content for c in split_docs
                if len(c.page_content.strip()) > min_content_len]
    # DEDUPLICATION
    prepFile = list(dict.fromkeys(prepFile))
    
    print(f"Chunks made: {len(prepFile)}")
    return prepFile


def get_model(model_name: str) -> SentenceTransformer:
    embedding_model = SentenceTransformer(model_name)
    return embedding_model


def emb_text(text: str, embedding_model: SentenceTransformer,
            norm: bool = True) -> list:
    return embedding_model.encode([text], normalize_embeddings=norm).tolist()[0]
