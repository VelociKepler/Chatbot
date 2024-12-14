import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Use ChatOpenAI for chat models
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Create an instance of the FastAPI application
app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],  # Allowed origins
    allow_credentials=True,  # Allow credentials (cookies, authorization headers, etc.)
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

def get_chain(llm, vectorstore, custom_template):
    prompt_template = PromptTemplate(template=custom_template, input_variables=["question", "context"])
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

def create_llm_chain(query, vectorstore_path, context):
    llm = ChatOpenAI(model_name="gpt-4o-mini")  # Use ChatOpenAI for chat models
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

    custom_template = f"""
    You are an AI assistant specializing in the Thai language name "น้องบอท" , serving customer information about product
    if user asking about product pls answer in markdouwn langurage like professional web bloger
    if you have to send product detail please send image too.
    Context: {{context}}
    Original question: {{question}}"""

    chain = get_chain(llm, vectorstore, custom_template)

    inputs = {
        "question": query,
        "context": context,
        "chat_history": []
    }

    result = chain.invoke(inputs)
    return result['answer']

def chatbot(query):
    try:
        logging.info(f"User: {query}")
        context = ""  
        response = create_llm_chain(query, "./yaml_vectorDB", context)
        return response

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return "ขออภัยครับ เกิดข้อผิดพลาดในระบบ กรุณาลองใหม่อีกครั้งในภายหลัง"

def main():
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
        response = chatbot(user_input)
        print(response)

class QueryRequest(BaseModel):
    question: str
    chat_history: list = []

@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    result = chatbot(request.question)
    return {"answer": result}