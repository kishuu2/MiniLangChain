"""A simple function to generate a summary of a PDF document using the Groq API."""

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
def generate_summary(pdf_path):

    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    prompt = PromptTemplate.from_template(
        "Summarize the following document:\n{document}"
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile")

    chain = prompt | llm

    result = chain.invoke({
        "document": pages[0].page_content[:1000]
    })

    return result.content