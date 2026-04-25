from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.texts = []
        self.index = None

    def build(self, texts):
        self.texts = texts
        embeddings = self.model.encode(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

    def search(self, query, k=3):
        q_emb = self.model.encode([query])
        D, I = self.index.search(np.array(q_emb), k)

        return [self.texts[i] for i in I[0]]
    