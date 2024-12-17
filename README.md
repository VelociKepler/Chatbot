# Thai AI Chatbot - FastAPI with LangChain and FAISS

## Overview
This project implements an AI Chatbot using FastAPI for serving API requests, LangChain for chain-based interaction, OpenAI GPT models for generating responses, and FAISS for fast and efficient similarity searches.

The chatbot, named "น้องบอท", is designed to assist in Thai language by serving product-related information and answering user queries professionally, styled like a web blogger with markdown formatting. The responses can include product images and links to product pages when applicable.

# Features
- FastAPI Backend: Serves a chatbot API endpoint.
- LangChain Integration: Uses ConversationalRetrievalChain to combine retrieval and generative responses.
- OpenAI GPT-4o Mini: Chat model for generating accurate, conversational answers.
- FAISS Vector Store: Efficiently stores and retrieves embeddings for document-based knowledge.
- Custom Prompting: Tailored prompts ensure the chatbot generates professional, markdown-style responses in Thai.
- CORS Middleware: Enables cross-origin resource sharing.

# Prerequisites
Ensure the following tools and libraries are installed on your system:

- Python 3.8+
- OpenAI API Key
- Required Python Libraries:
- fastapi
- langchain
- langchain_community
- langchain_openai
- uvicorn
- pydantic
- python-dotenv
- faiss-cpu
- A pre-built FAISS vectorstore stored locally in ./yaml2.(or whatever you set dir name for vectorDB)

# Installation
- Clone the Repository:
```
git clone <repository_url>
```

- Set up Environment Variables: Create a .env file in the root directory and add your OpenAI API Key:
```
OPENAI_API_KEY=your_openai_api_key
```

- Install Dependencies: Use the requirements.txt file to install the required libraries
```
pip install -r requirements.txt
```

- Run the Application: Start the FastAPI server using uvicorn:
```
uvicorn main:app --reload
```
By default, the application runs on http://127.0.0.1:8000.

# API Usage
- Endpoint: POST /api/chat
- Request Body:
```
{
  "question": "What products are available?",
  "chat_history": []
}
```
- Response
```
{
  "answer": "Here is a detailed product list with markdown formatting..."
}
```
# Notes
- Ensure the FAISS vectorstore is correctly placed in the ./yaml2 directory.
- Update OPENAI_API_KEY in your .env file.
- Modify the prompt template or model parameters in the main.py file for further customization.

# Author
Developed by [https://github.com/thiramet27].


