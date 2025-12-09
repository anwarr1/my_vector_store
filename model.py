class VectorRecord:
    def __init__(self, id, vector, text, metadata=None):
        self.id = id
        self.vector = vector  # list of floats
        self.text = text
        self.metadata = metadata or {}
    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata
        }

    @staticmethod
    def from_dict(d, vector):
        return VectorRecord(
            id=d["id"],
            vector=vector,
            text=d["text"],
            metadata=d["metadata"]
        )
    