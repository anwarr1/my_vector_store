from model import VectorRecord
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
import faiss  

class VectorStore:

    def __init__(self):
        self.records = []
        self.index = None  # FAISS index will be initialized later

    def add_record(self, id, vector, model, text, metadata=None):
        record = VectorRecord(id, vector, text, metadata)
        self.records.append(record)

    def build_faiss_index(self):
        if not self.records:
            raise ValueError("No records to build index.")

        dim = len(self.records[0].vector)
        self.index = faiss.IndexFlatL2(dim)  # exact L2 search
        vectors = np.array([record.vector for record in self.records], dtype="float32")
        self.index.add(vectors)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        # Save vectors
        vectors = np.array([record.vector for record in self.records], dtype="float32")
        np.save(os.path.join(directory, "vectors.npy"), vectors)

        # Save metadata + text
        json_records = [record.to_dict() for record in self.records]
        with open(os.path.join(directory, "records.json"), "w", encoding="utf-8") as f:
            json.dump(json_records, f)

        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, "faiss.index"))

    @classmethod
    def load(cls, directory):
        store = cls()
        # Load vectors
        vectors_path = os.path.join(directory, "vectors.npy")
        vectors = np.load(vectors_path)

        # Load metadata + text
        records_path = os.path.join(directory, "records.json")
        with open(records_path, "r", encoding="utf-8") as f:
            json_records = json.load(f)

        for rec_dict, vector in zip(json_records, vectors):
            record = VectorRecord.from_dict(rec_dict, vector)
            store.records.append(record)

        # Load FAISS index if exists
        faiss_path = os.path.join(directory, "faiss.index")
        if os.path.exists(faiss_path):
            store.index = faiss.read_index(faiss_path)
        else:
            store.build_faiss_index()  # fallback if no saved index

        print(f"Loaded {len(store.records)} records from {directory}")
        return store

    def search(self, query_vector, top_k=5, metadata_filter=None):
        if self.index is None:
            self.build_faiss_index()

        query_np = np.array(query_vector, dtype="float32").reshape(1, -1)
        distances, indices = self.index.search(query_np, top_k)

        results = []
        for idx in indices[0]:
            if idx == -1:  # no match
                continue
            record = self.records[idx]

            # Apply metadata filter
            if metadata_filter:
                if not all(record.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue

            results.append(record)
        return results

    @staticmethod
    def build_vector_store_from_documents(folder_path, chunk_size=500, overlap=50):
        store = VectorStore()
        model = SentenceTransformer("all-MiniLM-L6-v2")

        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    for i in range(0, len(text), chunk_size - overlap):
                        chunk = text[i:i + chunk_size]
                        if chunk:
                            vector = model.encode(chunk)
                            record_id = f"{filename}_{i}"
                            store.add_record(record_id, vector, None, chunk, {"source_file": filename})
        store.build_faiss_index()  # build FAISS after adding all chunks
        return store


if __name__ == "__main__":

    VECTOR_STORE_DIR = "vector_store"

    if os.path.exists(VECTOR_STORE_DIR):
        # Load existing store (fast, FAISS index loaded automatically)
        store = VectorStore.load(VECTOR_STORE_DIR)
    else:
        # Build new store from documents
        store = VectorStore.build_vector_store_from_documents("data/")
        store.save(VECTOR_STORE_DIR)

    loaded_store = VectorStore.load(VECTOR_STORE_DIR)
    query_text = "in what sectors ai is evolving?"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query_text])[0]

    results = loaded_store.search(query_embedding, top_k=3)
    for record in results:
        print(record.id)
        print("Found chunk:", record.text)
