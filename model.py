class VectorRecord:
    def __init__(self, id, vector, text, metadata=None):
        self.id = id
        self.vector = vector  # list of floats
        self.text = text
        self.metadata = metadata or {}
