from model import VectorRecord
from similarity import Similarity

class VectorStore:
      def __init__(self):
        self.records = [] 
      def add_record(self, id,vector,model,text ,metadata=None):
        record = VectorRecord(id, vector, text, metadata)
        self.records.append(record)
           
      def search(self, query_vector, top_k=5, metadata_filter=None):
         scored = []
     
         for record in self.records:
             # optional metadata filter
             if metadata_filter:
                 if not all(record.metadata.get(k) == v for k, v in metadata_filter.items()):
                     continue
     
             score = Similarity.cosine_similarity(query_vector, record.vector)
             scored.append((score, record))
     
         # sort by score descending (highest similarity first)
         scored.sort(key=lambda x: x[0], reverse=True)
     
         # return top-K records
         return [record for score, record in scored[:top_k]]


  