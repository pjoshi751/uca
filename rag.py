from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader('../docs/social-benefit-programs.pdf')
documents = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

vector_store = FAISS.from_documents(documents, embeddings)

query = "As a student do I get any benefit?"
results = vector_store.similarity_search(query)
for result in results:
    print(result.page_content)
