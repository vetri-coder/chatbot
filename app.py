from flask import Flask, render_template, request, jsonify
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Generation settings
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 65536,
    "response_mime_type": "text/plain",
}

# System prompt
system_prompt = """
Your job is to answer questions about the organization, its events, programs, membership, collaborations, and general user support.
Respond clearly, concisely, and helpfully. Your tone should be friendly, welcoming, and professional.
Use simple and clear English. Keep answers short (1–3 sentences) unless asked for details.
If the question is not related to Impacteers, respond with:
"I'm here to help with anything related to Impacteers! For other queries, please reach out to our team through the contact page."
"""

# Initialize model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config=generation_config,
    system_instruction=system_prompt.strip()
)

app = Flask(__name__)

# Homepage
@app.route('/')
def index():
    return render_template('index.html')

# API route
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(user_input)
    return jsonify({'response': response.text})

if __name__ == '__main__':
    app.run(debug=True)
