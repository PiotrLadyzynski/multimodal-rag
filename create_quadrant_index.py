import qdrant_client
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageDocument

from llama_index.core.schema import ImageNode


documents_images = SimpleDirectoryReader("./ikea_manuals/fredde/").load_data()+SimpleDirectoryReader("./ikea_manuals/smagoera/").load_data()+SimpleDirectoryReader("./ikea_manuals/tuffing/").load_data()

# Create a local Qdrant vector store
client = qdrant_client.QdrantClient(path="qdrant_index")

text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)
image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)
storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)

# Create the MultiModal index
index = MultiModalVectorStoreIndex.from_documents(
    documents_images,
    storage_context=storage_context,
)

# Save the index to disk
index.save_to_disk("index.json")

retriever_engine = index.as_retriever(image_similarity_top_k=2)