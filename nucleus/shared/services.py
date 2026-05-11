from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

load_dotenv()

host = os.getenv("VECTOR_DB_HOST")
port = int(os.getenv("VECTOR_DB_PORT"))
model_name = os.getenv("EMBEDDING_MODEL")

collection_long = os.getenv("COLLECTION_LONG")
collection_mid  = os.getenv("COLLECTION_MID")
vector_size     = int(os.getenv("VECTOR_SIZE"))

classifier_model = os.getenv("CLASSIFIER_MODEL")

class Services:
    def __init__(self):
        self.embedder = None
        self.vectordb = None

    async def initialize(self):
        print("Loading Embedding-Model...")
        self.embedder = SentenceTransformer(model_name)

        print("Loading Classifier...")
        self.classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model)
        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_model)
        self.classifier_model.eval()

        print("Connecting Vector Storage...")
        self.vectordb = QdrantClient(host=host, port=port)

        await self._ensure_collections()

        print("Services ready.")
    
    async def _ensure_collections(self):
        existing = [c.name for c in self.vectordb.get_collections().collections]

        for collection in (collection_long, collection_mid):
            self.vectordb = self.vectordb.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Collection {collection} initialized.")
        else:
            print(f"Collection {collection} found.")