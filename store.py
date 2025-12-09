from model import VectorRecord
from similarity import Similarity
import numpy as np
import os
import json 
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


if __name__ == "__main__":
    store = VectorStore()
    record1 = VectorRecord(id="1", vector=[0.1, 0.2, 0.3], text="First record", metadata={"category": "A"})
    record2 = VectorRecord(id="2", vector=[0.4, 0.5, 0.6], text="Second record", metadata={"category": "B"})

    store.add_record(record1.id, record1.vector, None, record1.text, record1.metadata)
    store.add_record(record2.id, record2.vector, None, record2.text, record2.metadata)

    # store.save("vector_store")
    loaded_store = VectorStore.load("vector_store")

    query_vec = [0.1, 0.2, 0.25]
    results = loaded_store.search(query_vec, top_k=1)
    for res in results:
        print(res.id, res.text)
