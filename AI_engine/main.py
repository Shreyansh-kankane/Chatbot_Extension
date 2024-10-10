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
from fastapi import FastAPI, HTTPException , UploadFile, File, Form


from pydantic import BaseModel
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from functools import lru_cache
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import chromadb
import PyPDF2

load_dotenv()

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize Chroma Cloud client
chroma_client = chromadb.Client()


# RAG class definition
class RAG:
    def __init__(self, namespace, model="gpt-4o-mini"):
        # Initialize the language model
        self.llm = ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))
        # Load the vector store for the specific website (namespace)
        self.vectorstore = chroma_client.get_or_create_collection(name=namespace)
        

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
    domain: str

# Mapping of website codes to namespaces (to be populated during setup)

# Mapping of domain names to namespaces (e.g., example.com -> namespace1)
# We store them on registering as a Website owner
domain_namespace = {} 


# Endpoint to initialize RAG based on a code (namespace)
@app.post("/init-rag")
async def init_rag(request: InitRequest):
    try:
        namespace = domain_namespace.get(request.domain)  # Retrieve the namespace using the provided code
        if not namespace:
            return {"error": "Invalid Code"}

        # Lazily load or get the cached RAG instance
        get_rag_instance(namespace)

        return {"message": f"RAG initialized for namespace: {namespace}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



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
    


# @app.post('/createEmbeddings')
# async def create_embeddings(namespace: str = Form(...), file: UploadFile = File(...)):
#     pdf_reader = PyPDF2.PdfReader(file.file)
    
#     print(os.getenv("OPENAI_API_KEY"))
#     for key, value in os.environ.items():
#         print(f"{key}: {value}")
#     file_text = ""
#     for page in pdf_reader.pages:
#         file_text += page.extract_text()

#     # Create embeddings using OpenAI
#     print(file_text)
#     try:
        
#         embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
#         embeddings = embeddings_model.embed_documents([file_text])

#         # Create or load a Chroma vector store for the given namespace
#         vector_store = Chroma(
#             persist_directory=f"./data/{namespace}",  # Namespace-based storage
#             embedding_function=embeddings_model
#         )

#         # Add embeddings to the vector store
#         vector_store.add_texts([file_text], embeddings)

#         # Persist the vector store for later use
#         vector_store.persist()

#         return {"message": "Embeddings created and stored successfully"}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

    
# @app.post("/test_query")
# async def ask_question(namespace : str  , query: QueryRequest):
#     try:
#         rag_instance = RAG(namespace)

#         # Generate the answer using the RAG engine
#         answer = rag_instance.generate_answer(query.question)
#         return {"question": query.question, "answer": answer}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
# Tasks ->  
# 1> Shift to pinecone
# 2> Integrate with backend Microservice
# 3> testing with cache api
# 4> Alternative to caching RAG instances ( possibly deploying seperate RAG chains on cloud for each website/namespace)
# 5> Start creating a dummy frontend for register/initialize , login , dashboard , chat component

# Installation ->
 
# Install all files from requirement.txt
# Run the application with: uvicorn filename:app --reload
