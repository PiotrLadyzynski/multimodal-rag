import qdrant_client
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageDocument

from llama_index.core.schema import ImageNode
import openai

documents_images = SimpleDirectoryReader("./ikea_manuals/fredde/").load_data()+SimpleDirectoryReader("./ikea_manuals/smagoera/").load_data()+SimpleDirectoryReader("./ikea_manuals/tuffing/").load_data()

OPENAI_API_TOKEN = ""
openai.api_key = OPENAI_API_TOKEN


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
#index.save_to_disk("index.json")

#retriever_engine = index.as_retriever(image_similarity_top_k=2)

#from llama_index.core.indices.multi_modal.retriever import (
#    MultiModalVectorIndexRetriever,
#)


query2 = "What parts are included in the Fredde?"
#assert isinstance(retriever_engine, MultiModalVectorIndexRetriever)
# retrieve for the query using text to image retrieval
#retrieval_results = retriever_engine.text_to_image_retrieve(query2)

#retrieved_images = []
#for res_node in retrieval_results:
#    if isinstance(res_node.node, ImageNode):
#        retrieved_images.append(res_node.node.metadata["file_path"])
#    else:
#        display_source_node(res_node, source_length=200)

#print(retrieval_results)
#print(retrieved_images)


from llama_index.core import PromptTemplate
from llama_index.core.query_engine import SimpleMultiModalQueryEngine


import os
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-turbo", max_new_tokens=300
)

qa_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_tmpl = PromptTemplate(qa_tmpl_str)

query_engine = index.as_query_engine(
    llm=openai_mm_llm, text_qa_template=qa_tmpl
)

#query3 = "What are the steps to assemble Fredde?"
query3 = "What parts are included in the Fredde?"

response = query_engine.query(query3)

print(query3)
print("==== Chat GPT response =====")

print(response)

print([n.metadata["file_path"] for n in response.metadata["image_nodes"]])


retriever_engine = index.as_retriever(
    similarity_top_k=2, image_similarity_top_k=2
)
# retrieve more information from the GPT4V response
retrieval_results = retriever_engine.retrieve(query3)

print("==== Retriever response  =====")

print(str(retrieval_results))







query3 = "What is step 4 of assembling the Fredde?"

response = query_engine.query(query3)

print(query3)
print("==== Chat GPT response =====")

print(response)

print([n.metadata["file_path"] for n in response.metadata["image_nodes"]])


retriever_engine = index.as_retriever(
    similarity_top_k=2, image_similarity_top_k=2
)
# retrieve more information from the GPT4V response
retrieval_results = retriever_engine.retrieve(query3)

print("==== Retriever response  =====")

print(str(retrieval_results))






