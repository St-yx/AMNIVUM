import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

host = os.getenv("VECTOR_DB_HOST")
port = int(os.getenv("VECTOR_DB_PORT"))
model_name = os.getenv("EMBEDDING_MODEL")

collection_long = os.getenv("COLLECTION_LONG", "memoria_long")
collection_mid  = os.getenv("COLLECTION_MID", "memoria_mid")
vector_size     = int(os.getenv("VECTOR_SIZE"))

classifier_model = os.getenv("CLASSIFIER_MODEL")

class Services:
    def __init__(self):
        self.embedder = None
        self.vecdb = None

    async def initialize(self):
        print("Loading Embedding-Model...")
        self.embedder = SentenceTransformer(model_name)

        print("Loading Classifier...")
        self.classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model)
        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_model)
        self.classifier_model.eval()

        print("Connecting Vector Storage...")
        self.vecdb = QdrantClient(host=host, port=port)

        await self._ensure_collections()

        print("Services ready.")
    
    async def _ensure_collections(self):
        existing = [c.name for c in self.vecdb.get_collections().collections]

        for collection in (collection_long, collection_mid):
            if collection in existing:
                print(f"Collection {collection} found.")
                continue
            self.vecdb.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Collection {collection} initialized.")