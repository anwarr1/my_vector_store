class Similarity:

    @staticmethod
    def dot(a, b):
        # sum(a[i] * b[i])
        total = 0
        for i in range(len(a)):
            total += a[i] * b[i]
        return total

    @staticmethod
    def norm(v):
        # sqrt(sum(v[i]^2))
        total = 0
        for i in range(len(v)):
            total += v[i] * v[i]
        return total ** 0.5

    @staticmethod
    def cosine_similarity(a, b):
        # dot(a, b) / (norm(a) * norm(b))
        return Similarity.dot(a, b) / (Similarity.norm(a) * Similarity.norm(b))

    
if __name__ == "__main__":
    a = [1, 0]
    b = [1, 0]
    print(Similarity.cosine_similarity(a, b))