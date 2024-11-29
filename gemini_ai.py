import google.generativeai as genai
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') # st.secrets['GEMINI_API_KEY']

genai.configure(api_key = GEMINI_API_KEY)

model_gemini_pro = genai.GenerativeModel(
    'gemini-1.5-flash', 
    system_instruction='Вы профессиональный ассистент кардиолога. Вы помогаете с анализом результатов клинических данных пациента и даете рекомендации.'
)

def generate_ai_response(prompt: str) -> str:
    response = model_gemini_pro.generate_content(prompt)
    return response.text
