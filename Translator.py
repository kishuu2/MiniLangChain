from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=1024
)

prompt = PromptTemplate.from_template("""
Translate the following text from {source_language} to {target_language}.

Text:
{input}

Return only the translated text.
""")

chain = prompt | llm

def translate_text(text, source_language, target_language):

    result = chain.invoke({
        "input": text,
        "source_language": source_language,
        "target_language": target_language
    })

    return result.content