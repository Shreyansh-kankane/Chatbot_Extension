# import os
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from langchain import hub
# from langchain_openai import ChatOpenAI
# from langchain_chroma import Chroma
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import OpenAIEmbeddings

# # Set the OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# # Store RAG engines for multiple websites (by namespace)
# rag_engines = {}

# # RAG class definition
# class RAG:
#     def __init__(self, namespace, model="gpt-4o-mini"):
#         # Initialize the language model
#         self.llm = ChatOpenAI(model=model)

#         # Load the vector store for the specific website (namespace)
#         self.vectorstore = Chroma(
#             embedding=OpenAIEmbeddings(),
#             persist_directory=f"chroma_db/{namespace}"  # Each website/namespace has its own storage
#         )

#         self.retriever = self.vectorstore.as_retriever()
#         self.prompt = hub.pull("rlm/rag-prompt")

#         # RAG chain definition
#         self.rag_chain = (
#             {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
#             | self.prompt
#             | self.llm
#             | StrOutputParser()
#         )

#     def format_docs(self, docs):
#         """Format retrieved documents into a readable string format."""
#         return "\n\n".join(doc.page_content for doc in docs)

#     def generate_answer(self, question):
#         """Generate an answer to the query using the RAG pipeline."""
#         return self.rag_chain.invoke(question)

# # FastAPI app initialization
# app = FastAPI()

# # Model for the request to ask questions
# class QueryRequest(BaseModel):
#     question: str

# # Model for the initialization request
# class InitRequest(BaseModel):
#     code: str

# # Mapping of website codes to namespaces (to be populated during setup)
# code_mp = {}

# # Endpoint to initialize RAG based on a code (namespace)
# @app.post("/init-rag")
# async def init_rag(request: InitRequest):
#     try:
#         namespace = code_mp.get(request.code)  # Retrieve the namespace using the provided code
#         if not namespace:
#             return {"error": "Invalid Code"}

#         # Initialize RAG for this namespace if not already initialized
#         if namespace not in rag_engines:
#             rag_engines[namespace] = RAG(namespace=namespace)

#         return {"message": f"RAG initialized for namespace: {namespace}"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Endpoint to ask questions and get answers from RAG
# domain_namespace = {}

# @app.post("/query")
# async def ask_question(domain : str , query: QueryRequest):
#     try:
#         namespace = domain_namespace.get(domain)
#         # Use the corresponding RAG engine for the website/namespace to generate the answer
#         if namespace not in rag_engines:
#             rag_engines[namespace] = RAG(namespace=namespace)

#         # Generate the answer using the RAG engine
#         answer = rag_engines[namespace].generate_answer(query.question)
#         return {"question": query.question, "answer": answer}
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# Run the application with: uvicorn filename:app --reload



import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from functools import lru_cache

# Set the OpenAI API key

# RAG class definition
class RAG:
    def __init__(self, namespace, model="gpt-4o-mini"):
        # Initialize the language model
        self.llm = ChatOpenAI(model=model)

        # Load the vector store for the specific website (namespace)
        self.vectorstore = Chroma(
            embedding=OpenAIEmbeddings(),
            persist_directory=f"chroma_db/{namespace}"  # Each website/namespace has its own storage
        )

        self.retriever = self.vectorstore.as_retriever()
        self.prompt = hub.pull("rlm/rag-prompt")

        # RAG chain definition
        self.rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_answer(self, question):
        return self.rag_chain.invoke(question)

# Lazy loading and caching of RAG instances
@lru_cache(maxsize=100)  # Cache up to 100 RAG instances
def get_rag_instance(namespace: str):
    """Lazy-load the RAG instance, caching the most recently used instances."""
    return RAG(namespace=namespace)

# FastAPI app initialization
app = FastAPI()

# Model for the request to ask questions
class QueryRequest(BaseModel):
    question: str

# Model for the initialization request
class InitRequest(BaseModel):
    code: str

# Mapping of website codes to namespaces (to be populated during setup)
code_mp = {}

# Endpoint to initialize RAG based on a code (namespace)
@app.post("/init-rag")
async def init_rag(request: InitRequest):
    try:
        namespace = code_mp.get(request.code)  # Retrieve the namespace using the provided code
        if not namespace:
            return {"error": "Invalid Code"}

        # Lazily load or get the cached RAG instance
        get_rag_instance(namespace)

        return {"message": f"RAG initialized for namespace: {namespace}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mapping of domain names to namespaces (e.g., example.com -> namespace1)
domain_namespace = {}

# Endpoint to ask questions and get answers from RAG
@app.post("/query")
async def ask_question(domain: str, query: QueryRequest):
    try:
        namespace = domain_namespace.get(domain)
        if not namespace:
            raise HTTPException(status_code=404, detail="Namespace not found")

        # Lazily load or get the cached RAG instance
        rag_instance = get_rag_instance(namespace)

        # Generate the answer using the RAG engine
        answer = rag_instance.generate_answer(query.question)
        return {"question": query.question, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application with: uvicorn filename:app --reload
