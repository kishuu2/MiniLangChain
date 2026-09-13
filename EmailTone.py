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

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is not configured")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key
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

    data = request.get_json(silent=True) or {}

    email_text = data.get("message")
    tone = data.get("tone")
    if not email_text or not tone:
        return jsonify({"error": "message and tone are required"}), 400

    try:
        result = chain.invoke({
            "input": email_text,
            "tone": tone
        })
    except Exception as error:
        app.logger.exception("Email rewrite failed")
        return jsonify({"error": str(error)}), 502

    return jsonify({
        "response": result.content
    })

@app.route("/summarize", methods=["POST"])
def summarize():
    file = request.files.get('file')
    if not file or not file.filename:
        return jsonify({"error": "A PDF file is required"}), 400
    file_path = "temp.pdf"

    try:
        file.save(file_path)
        summary = generate_summary(file_path)
    except Exception as error:
        app.logger.exception("Document summarization failed")
        return jsonify({"error": str(error)}), 502
    return jsonify({
        "summary": summary
    })

@app.route("/translate", methods=["POST"])
def translate():

    data = request.get_json(silent=True) or {}

    text = data.get("text")
    source_language = data.get("source_language")
    target_language = data.get("target_language")
    if not text or not source_language or not target_language:
        return jsonify({"error": "text, source_language and target_language are required"}), 400

    try:
        translated = translate_text(
            text,
            source_language,
            target_language
        )
    except Exception as error:
        app.logger.exception("Translation failed")
        return jsonify({"error": str(error)}), 502

    return jsonify({
        "translation": translated
    })

@app.route("/health", methods=["POST"])
def health():

    data = request.get_json(silent=True) or {}

    symptoms = data.get("symptoms")
    if not symptoms:
        return jsonify({"error": "symptoms are required"}), 400

    try:
        response = health_advice(symptoms)
    except Exception as error:
        app.logger.exception("Health advice request failed")
        return jsonify({"error": str(error)}), 502

    return jsonify({
        "advice": response
    })
if __name__ == "__main__":
    app.run(debug=True)