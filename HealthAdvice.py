from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
load_dotenv()

memory = ConversationBufferMemory()

llm = ChatGroq(model="openai/gpt-oss-20b", max_tokens=512)
prompt = PromptTemplate.from_template("""
You are a cautious health information assistant. Give a helpful answer in 2 or 3
sentences. Do not diagnose or prescribe medicine. Provide general health tips and
recommend consulting a qualified doctor when appropriate.

Conversation and current symptoms:
{symptoms}
""")

chain = prompt | llm

def health_advice(symptoms):

    history = memory.load_memory_variables({})["history"]

    history += f"\nYou: {symptoms}\nExpert: "

    result = chain.invoke({
        "symptoms": history
    })

    memory.save_context(
        {"input": symptoms},
        {"output": result.content}
    )

    response = result.content.strip() if isinstance(result.content, str) else ""
    if not response:
        raise RuntimeError("The health model returned an empty response")

    return response
    