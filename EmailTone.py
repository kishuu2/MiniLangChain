"""A simple Flask application that uses the Groq API to rewrite emails in a more formal tone. The application has a single endpoint that accepts a POST request with the email text and the desired tone, and returns the rewritten email."""
from flask import Flask, render_template, request, jsonify
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from DocumentSummary import generate_summary
from Translator import translate_text
from HealthAdvice import health_advice
import os
from flask_cors import CORS
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
import os

load_dotenv()
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

prompt = PromptTemplate.from_template(
"""
Rewrite the following email in a more formal tone:

{input}

Desired tone: {tone}
"""
)

chain = prompt | llm


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/rewrite", methods=["POST"])
def rewrite():

    data = request.json

    email_text = data.get("message")
    tone = data.get("tone")

    result = chain.invoke({
        "input": email_text,
        "tone": tone
    })

    return jsonify({
        "response": result.content
    })

@app.route("/summarize", methods=["POST"])
def summarize():
    file = request.files['file']
    file_path = "temp.pdf"

    file.save(file_path)

    summary = generate_summary(file_path)
    return jsonify({
        "summary": summary
    })

@app.route("/translate", methods=["POST"])
def translate():

    data = request.get_json()

    text = data.get("text")
    source_language = data.get("source_language")
    target_language = data.get("target_language")

    translated = translate_text(
        text,
        source_language,
        target_language
    )

    return jsonify({
        "translation": translated
    })

@app.route("/health", methods=["POST"])
def health():

    data = request.get_json()

    symptoms = data.get("symptoms")

    response = health_advice(symptoms)

    return jsonify({
        "advice": response
    })
if __name__ == "__main__":
    app.run(debug=True)