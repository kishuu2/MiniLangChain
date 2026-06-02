from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
load_dotenv()

memory = ConversationBufferMemory()

llm = ChatGroq(model="llama-3.3-70b-versatile", max_tokens=100)
prompt = PromptTemplate.from_template("give the answer in short, maximum 2 or 3 sentences and you are an expert in health advice: {symptoms}. Do not diagnose. Provide general health tips only. Suggest consulting a doctor.")

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

    return result.content
    