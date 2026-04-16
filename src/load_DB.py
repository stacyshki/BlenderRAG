from chromadb import PersistentClient
from tqdm import tqdm


def addInitDB(path: str, collection_name: str,
                embedding_func, embedding_file: list,
                batch_size: int):
    
    client = PersistentClient(path=path)
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception:
        print(f"Creating new collection '{collection_name}'...")
        collection = client.create_collection(name=collection_name)
        
        embeddings = []
        ids = []
        
        for i, line in enumerate(tqdm(embedding_file,
                                    desc="Creating embeddings")):
            embeddings.append(embedding_func(line))
            ids.append(str(i))
        
        for batch_start in tqdm(range(0, len(embedding_file), batch_size),
                                desc="Adding to DB"):
            batch_end = batch_start + batch_size
            collection.add(
                documents=embedding_file[batch_start:batch_end],
                embeddings=embeddings[batch_start:batch_end],
                ids=ids[batch_start:batch_end]
            )
        
        print(f"Indexed {len(ids)} chunks.")
    
    return collection
