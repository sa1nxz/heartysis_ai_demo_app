import streamlit as st
import pandas as pd
import numpy as np
from gemini_ai import generate_ai_response
import joblib

st.set_page_config(
    page_title="Heartysis AI: Прогнозируем сердечные заболевания", 
    page_icon="♥", 
    layout="wide"
)

@st.cache_resource()
def load_model():
    with open('models/ensemble_model.joblib', 'rb') as file:
        model = joblib.load(file)
    print("Model loaded successfully!")
    return model

loaded_model = load_model()

def preprocess_input(age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope):
    def transform_Sex(sex):
        return 1 if sex == 'М' else 0

    def transform_ChestPainType(cpt):
        if cpt == 'Атипичная стенокардия': return 0
        elif cpt == 'Боль, не связанная со стенокардией': return 1
        elif cpt == 'Бессимптомный': return 2
        else: return 3

    def transform_RestingECG(recg):
        if recg == 'Норма': return 0
        elif recg == 'Аномалия зубца ST-T': return 1
        else: return 2
        
    def transform_ExerciseAngina(ea):
        return 1 if ea == 'Да' else 0

    def transform_ST_Slope(sts):
        if sts == 'Плоский': return 0
        elif sts == 'Вверх': return 1
        else: return 2    
    
    sex = transform_Sex(sex)
    chest_pain_type = transform_ChestPainType(chest_pain_type)
    fasting_bs = 1 if fasting_bs == '> 120 мг/дл' else 0
    resting_ecg = transform_RestingECG(resting_ecg)
    exercise_angina = transform_ExerciseAngina(exercise_angina)
    st_slope = transform_ST_Slope(st_slope)

    return np.array([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

def predict_heart_disease_probability(input_data):
    try:
        probability = loaded_model.predict_proba(input_data)[:, 1]
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None
    return probability


st.title(":heart: Heartysis AI: Прогнозируем сердечные заболевания")
st.write("Это веб-приложение предсказывает наличие сердечных заболеваний у пациента на основе клинических данных.")

colA, colB = st.columns([2, 5])

with colA:
    st.markdown('''
        <img src="https://www.peninsulaheartclinic.co.uk/wp-content/uploads/2019/10/NEW_Heartbeat_looping_GIF_NORMAL_0.gif" width="350">
    ''', unsafe_allow_html=True)

with colB:
    st.markdown('''
        #### Что такое сердечные заболевания?
        Сердечно-сосудистые заболевания (ССЗ) — это группа заболеваний, связанных с нарушением работы сердца и сосудов.

        #### Как работает модель?
        Модель принимает входные данные пациента и предсказывает вероятность наличия сердечного заболевания.
    ''')

st.header("Введите информацию о пациенте")
with st.container():
    age = st.number_input("Возраст", min_value=1, max_value=120)
    sex = st.radio("Пол", ["М", "Ж"])
    chest_pain_type = st.selectbox("Тип боли в груди", ["Типичная стенокардия", "Атипичная стенокардия", "Боль, не связанная со стенокардией", "Бессимптомный"])
    resting_bp = st.number_input("Кровяное давление в состоянии покоя (мм рт.ст.)")
    cholesterol = st.number_input("Уровень холестерина в сыворотке крови (мм/дл)")
    fasting_bs = st.radio("Уровень сахара в крови натощак", ["> 120 мг/дл", "<= 120 мг/дл"])
    resting_ecg = st.selectbox("Результаты электрокардиограммы в состоянии покоя", ["В норме", "ST", "ГЛЖ"])
    max_hr = st.number_input("Достигнута максимальная частота сердечных сокращений")
    exercise_angina = st.radio("Стенокардия, вызванная физической нагрузкой", ["Да", "Нет"])
    oldpeak = st.number_input("Пиково-Низкое значение (ST депрессия)")
    st_slope = st.selectbox("ST Slope", ["Вверх", "Плоский", "Вниз"])

if st.button("Анализ"):
    with st.spinner("Анализируем.."):
        preprocessed_input = preprocess_input(age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope)
        heart_disease_probability = predict_heart_disease_probability(preprocessed_input)
        st.write("Вероятность сердечных заболеваний:", round(heart_disease_probability[0], 4))
        if heart_disease_probability[0] > 0.5:
            st.warning(":broken_heart: У пациента, скорее всего, есть заболевание сердца.")
        else:
            st.success(":green_heart: Маловероятно, что у пациента есть сердечные заболевания.")

st.write("---")

# Heartysis AI Chat Assistant
st.subheader("🩺 Heartysis AI Chat Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Gemini AI user response
if prompt := st.chat_input("Чем я могу вам помочь?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    try:
        ai_answer = generate_ai_response(prompt)
    except Exception as e:
        ai_answer = f'Произошла ошибка во время генерации ответа: {e}'

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        st.markdown(ai_answer)

    st.session_state.messages.append({"role": "assistant", "content": ai_answer})