from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0.1,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
system_prompt = "You are an expert in translating English to Arabic. Please provide accurate translations while maintaining the original meaning and context."
output_parser = StrOutputParser()
# System + User Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{query}")
    ]
)
chain = prompt | llm | output_parser

app = FastAPI(title="English to Arabic Translation API", version="1.0",
              description="An API that translates English text to Arabic using a language model.")

add_routes(app, chain, path="/translate")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
 
 # Invoke chain
# response = chain.invoke({
#     "query": "Explain attention mechanism in transformers"
# })