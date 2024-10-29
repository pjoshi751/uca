from haystack import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import PDFToTextConverter, PreProcessor
from haystack.nodes import EmbeddingRetriever

faiss_index_path = "faiss_index_test"
faiss_config_path = "faiss_config_test"

document_store = FAISSDocumentStore(embedding_dim=384)
converter = PDFToTextConverter()
preprocessor = PreProcessor()

indexing_pipeline = Pipeline()
indexing_pipeline.add_node(component=converter, name="PDFConverter", inputs=["File"])
indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["PDFConverter"])
indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PreProcessor"])

indexing_pipeline.run(file_paths=["social-benefit-programs.pdf"])


retriever = EmbeddingRetriever(
    document_store=document_store,
   embedding_model="sentence-transformers/all-MiniLM-L6-v2",
   model_format="sentence_transformers"
)
document_store.update_embeddings(retriever)

document_store.save(faiss_index_path, faiss_config_path)

