from model import VectorRecord
from similarity import Similarity
import numpy as np
import os
import json 
from sentence_transformers import SentenceTransformer


class VectorStore:

    def __init__(self):
        self.records = [] 

    def add_record(self, id,vector,model,text ,metadata=None):
        record = VectorRecord(id, vector, text, metadata)
        self.records.append(record)

    def save(self,directory):
        os.makedirs(directory, exist_ok=True)
        vectors=np.array([record.vector for record in self.records])
        np.save(os.path.join(directory, "vectors.npy"), vectors)

        json_records=[record.to_dict() for record in self.records]

        with open(os.path.join(directory, "records.json"), "w") as f:
            json.dump(json_records, f)

    def search(self, query_vector, top_k=5, metadata_filter=None):
        scored = []

        for record in self.records:
            # optional metadata filter
            if metadata_filter:
                if not all(
                    record.metadata.get(k) == v for k, v in metadata_filter.items()
                ):
                    continue

            score = Similarity.cosine_similarity(query_vector, record.vector)
            scored.append((score, record))

        # sort by score descending (highest similarity first)
        scored.sort(key=lambda x: x[0], reverse=True)

        # return top-K records
        return [record for score, record in scored[:top_k]]


    @classmethod
    def load(cls, directory):
        vectors_path = os.path.join(directory, "vectors.npy")
        records_path = os.path.join(directory, "records.json")

        # Load vectors
        vectors = np.load(vectors_path)

        # Load metadata + text
        with open(records_path, "r", encoding="utf-8") as f:
            json_records = json.load(f)

        # Create new store
        store = cls()
        for rec_dict, vector in zip(json_records, vectors):
            record = VectorRecord.from_dict(rec_dict, vector)
            store.records.append(record)

        print(f"Loaded {len(store.records)} records from {directory}")
        return store
    def build_vector_store_from_documents(folder_path, chunk_size=500, overlap=50):
        store=VectorStore()
        model = SentenceTransformer("all-MiniLM-L6-v2")

        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    for i in range(0, len(text), chunk_size - overlap):
                        chunk = text[i:i + chunk_size]
                        if chunk:
                            vector = model.encode(
                                chunk
                            )  # Placeholder for actual embedding
                            record_id = f"{filename}_{i}"
                            store.add_record(record_id, vector, None, chunk, {"source_file": filename})
        return store


if __name__ == "__main__":

    store = VectorStore.build_vector_store_from_documents("data/")
    store.save("vector_store")
    loaded_store = VectorStore.load("vector_store")
    query_text = "in what sectors ai is evolving?"
    query_embedding = SentenceTransformer("all-MiniLM-L6-v2").encode([query_text])[0]
    results = loaded_store.search(query_embedding, top_k=3)
    for record in results:
        print(record.id)
        print("Found chunk:", record.text)
