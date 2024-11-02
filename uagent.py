# Full end2end question/answer app

import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    num_thread=4,  # Change as per your CPU configuration
)

vector_store = FAISS.load_local(
    "../faiss/programs_index", embeddings, allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

tool = create_retriever_tool(
    retriever,
    "programs_retriever",
    "Searches and returns excerpts from the available programs",
)
tools = [tool]

system_prompt = '''Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Answer as best as possible on eligibility of programs, and how to apply.'''

agent_executor = create_react_agent(llm, tools, state_modifier=system_prompt)

#query = "As a mother am I eligible for some progams?"
query = "How is the weather today?"

n = 0
for event in agent_executor.stream(
    {"messages": [HumanMessage(content=query)]},
    stream_mode="values",
):
    n += 1
    if n==3:
        print(event)
    #print(event["messages"][-1].pretty_print())

