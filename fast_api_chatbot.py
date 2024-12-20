from fastapi import FastAPI, HTTPException
from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from pydantic import BaseModel
from pydantic import ValidationError
import os
import warnings
import uvicorn


# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._migration")


app = FastAPI()

API_KEY = "overt_chatbot_api_call"  # Ensure this matches the key used in curl

def api_key_dependency(api_key: str = Header(..., alias="x-api-key")) -> str:
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return api_key


class QueryRequest(BaseModel):
    question: str

def read_csv(file_path: str):
    loader = CSVLoader(file_path=file_path)
    docs = loader.load()
    return docs

def vector_stores_and_embeddings_csv(docs):
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    return db

file_path = "uploaded_csv/chennai.csv"
docs = read_csv(file_path)
db = vector_stores_and_embeddings_csv(docs)

retriever = db.as_retriever()
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

@app.get("/")
def read_root():
    return {"message": "API is running. Use the /gethelp endpoint to interact."}

@app.post("/gethelp")
async def query_csv(request: QueryRequest):#, api_key: str = Depends(api_key_dependency)):

    try:

        qa_chain_mr = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="refine"
        )

        # db = FAISS.from_documents(docs, embeddings)

        # db = Chroma.from_documents(documents=docs, embedding=embeddings)

        result = qa_chain_mr.invoke(request.question)

        return {"answer": result}


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8094)

#curl -X POST "http://127.0.0.1:8000/gethelp" -H "accept: application/json" -H "Content-Type: application/json" -H "x-api-key: overt_chatbot_api_call" -d "{\"question\": \"Name some beaches in Chennai\"}"

#curl -X POST "http://20.174.169.225:8094/gethelp" -H "accept: application/json" -H "Content-Type: application/json" -H "x-api-key: overt_chatbot_api_call" -d "{\"question\": \"Name some beaches in Chennai\"}"

#curl -X POST "http://20.174.169.225:8094/gethelp_vijayawada" -H "accept: application/json" -H "Content-Type: application/json" -H "x-api-key: overt_chatbot_api_call" -d "{\"question\": \"Name some beaches in Chennai\"}"


# curl -X POST "http://127.0.0.1:8000/gethelp" \
#   -H "accept: application/json" \
#   -H "Content-Type: application/json" \
#   -H "x-api-key: overt_chatbot_api_call" \
#   -d '{"question": "Name some beaches in Chennai"}'

#uvicorn tries_2:app --reload - Command to run the program
