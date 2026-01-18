from langchain_community.embeddings import HuggingFaceEmbeddings

class USEEmbeddings:
    def __init__(self):
        try:
            import tensorflow_hub as hub
        except ImportError as e:
            raise ImportError(
                "tensorflow_hub is required for USEEmbeddings but failed to load"
            ) from e

        self.model = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder/4"
        )

    def embed_documents(self, texts):
        return self.model(texts).numpy().tolist()

    def embed_query(self, text):
        return self.model([text]).numpy()[0].tolist()


def load_embeddings():
    embeddings = {
        "minilm": HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
        "mpnet": HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
    }

    # USE is optional
    try:
        embeddings["use"] = USEEmbeddings()
    except Exception as e:
        print("⚠️ USE embeddings not available:", str(e))

    return embeddings
