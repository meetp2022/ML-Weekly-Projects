from langchain_community.embeddings import HuggingFaceEmbeddings
import tensorflow_hub as hub

class USEEmbeddings:
    def __init__(self):
        self.model = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder/4"
        )

    def embed_documents(self, texts):
        return self.model(texts).numpy().tolist()

    def embed_query(self, text):
        return self.model([text]).numpy()[0].tolist()

def load_embeddings():
    return {
        "minilm": HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
        "mpnet": HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
        "use": USEEmbeddings()
    }
