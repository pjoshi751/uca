# Full end2end question/answer app
# Experimenation 

import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

'''
Print final response from AI 
'''
def print_ai_response(agent_executor, query, config):
    for event in agent_executor.stream({"messages": [HumanMessage(content=query)]}, config, stream_mode='values'):
        pass
    print(event['messages'][-1].pretty_repr())

def create_faiss_retriever_tool(model, docs_path, tool_name, tool_desc):
    embeddings = HuggingFaceEmbeddings(model_name=model)
    vector_store = FAISS.load_local(docs_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    tool = create_retriever_tool(retriever, tool_name, tool_desc)
    return tool

def load_llama(model, nthreads=4):
    llm = ChatOllama(
        model=model,
        temperature=0,
        num_thread=nthreads,  # Change as per your CPU configuration
    )
    return llm

