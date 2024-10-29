from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import PDFToTextConverter, PreProcessor
from haystack.nodes import EmbeddingRetriever

faiss_index_path = "faiss_index_test"
faiss_config_path = "faiss_config_test"

document_store = FAISSDocumentStore.load(
    index_path=faiss_index_path,
    config_path=faiss_config_path)

print(document_store.get_all_documents())
