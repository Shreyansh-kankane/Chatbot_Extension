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
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import chromadb
import PyPDF2
from langchain.prompts import PromptTemplate
from openai import OpenAI

import uvicorn

load_dotenv()

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Initialize Chroma Cloud client
chroma_client = chromadb.Client()


# RAG class definition
class RAG:
    def __init__(self, namespace="testing", model="gpt-4o-mini"):
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

class CustomEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, input):
        # Assuming input is a list of strings to be embedded
        return self.model.encode(input, convert_to_tensor=True)
    
    def embed_query(self, input):
        # Assuming input is a list of strings to be embedded
        return self.model.encode(input, convert_to_tensor=True)


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
    


@app.post('/createEmbeddings')
async def create_embeddings(namespace: str = Form(...), file: UploadFile = File(...)):
    pdf_reader = PyPDF2.PdfReader(file.file)
    
    print(os.getenv("OPENAI_API_KEY"))
    for key, value in os.environ.items():
        print(f"{key}: {value}")
    file_text = ""
    for page in pdf_reader.pages:
        file_text += page.extract_text()

    # Create embeddings using OpenAI
    print("Text Data-----------------" , file_text)
    try:
        
        # embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        # embeddings = embeddings_model.embed_documents([file_text])
        
        
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        custom_embeddings = CustomEmbeddingFunction(embeddings_model)
        persist_directory =  f'./data/{namespace}'
        
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=custom_embeddings)

        # Add the document to the vector store
        vector_store.add_texts(texts=[file_text], ids=[namespace])
        
        print("Created embeddings at /data/", namespace)
        return {"message": "Embeddings created and stored successfully "}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

def generate_response(context: str, query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a knowledgeable programming assistant."},
            {
                "role": "user",
                "content": (
                    f"Context: {context}\n"
                    f"Question: {query}"
                ),
            },
        ],
    )

    # Access the generated answer
    answer = response['choices'][0]['message']['content']
    return answer


    
@app.post("/test_query")
async def ask_question(namespace: str, query: QueryRequest):
    try:
        # Initialize the RAG instance for the specified namespace
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        custom_embeddings = CustomEmbeddingFunction(embeddings_model)
        
        db3 = Chroma(persist_directory=f"./data/{namespace}" , embedding_function=custom_embeddings)
        
        context = db3.similarity_search(query.question)
        
        if not context:
            return {"question": query.question, "answer": "No relevant information found."}
        

        # Combine the results into one string for clarity
        context = " ".join(result.page_content for result in context)
        
        
        print("Context----------" , context)

        # Use OpenAI or another model to generate a structured answer based on the context
        answer = generate_response(context, query.question)

        # Return the question and generated answer
        return {"question": query.question, "answer": answer}
        

        # Return the question and generated answer
        # return {"question": query.question, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="debug", debug=True)

