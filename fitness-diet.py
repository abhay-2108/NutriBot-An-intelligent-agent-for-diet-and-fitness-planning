import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings

import joblib
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

st.set_page_config(page_title="NutriBot", page_icon="🤖", layout="wide")
st.title("NutriBot : An Intelligent Agent for Diet and Fitness Planning")


model = OllamaLLM(model="llama2:13b", temperature=0.2, max_tokens=1000)

# Load pre-trained models
daily_model = joblib.load("model/daily-df1.pkl")
calories_model = joblib.load("model/calories-df2.pkl")
steps_model = joblib.load("model/steps-df3.pkl")
heart_rate_model = joblib.load("model/heartrate-df5.pkl")
sleep_model = joblib.load("model/sleep-df17.pkl")
weight_model = joblib.load("model/weight-BMI-df18.pkl")


st.sidebar.header("👤 Enter Your Fitness Details")

steps = st.sidebar.number_input("🦶 Daily Steps", min_value=0, max_value=36019, value=5000)
distance = st.sidebar.number_input("📏 Daily Distance (km)", min_value=0.0, max_value=28.0, value=5.0)
calories = st.sidebar.number_input("🔥 Daily Calories Consumed", min_value=0, max_value=4900, value=2000)

st.sidebar.markdown("### 🕒 Activity Breakdown (in Minutes)")
sedentary = st.sidebar.number_input("🪑 Sedentary Minutes", min_value=0, max_value=1440)
lightly_active = st.sidebar.number_input("🚶 Lightly Active Minutes", min_value=0, max_value=518)
fairly_active = st.sidebar.number_input("🏃 Fairly Active Minutes", min_value=0, max_value=143)
very_active = st.sidebar.number_input("🏋️ Very Active Minutes", min_value=0, max_value=210)

heart_rate = st.sidebar.number_input("❤️ Avg Heart Rate (bpm)", min_value=60, max_value=100)
sleep_hours = st.sidebar.number_input("😴 Avg Sleep (Minutes)", min_value=58.0, max_value=796.0)
weight = st.sidebar.number_input("⚖️ Weight (kg)", min_value=53.0, max_value=133.5)
bmi = st.sidebar.number_input("📉 BMI", min_value=22.0, max_value=50.0)


st.header("💡 Get Your Personalized Diet and Fitness Plan")


if st.button("Generate Plan"):
    st.success("Processing your inputs...")

    steps_scaled = scaler.fit_transform([[steps]])[0][0]
    distance_scaled = scaler.fit_transform([[distance]])[0][0]
    calories_scaled = scaler.fit_transform([[calories]])[0][0]
    sedentary_scaled = scaler.fit_transform([[sedentary]])[0][0]
    lightly_active_scaled = scaler.fit_transform([[lightly_active]])[0][0]
    fairly_active_scaled = scaler.fit_transform([[fairly_active]])[0][0]
    very_active_scaled = scaler.fit_transform([[very_active]])[0][0]
    heart_rate_scaled = scaler.fit_transform([[heart_rate]])[0][0]
    sleep_hours_scaled = scaler.fit_transform([[sleep_hours]])[0][0]
    weight_scaled = scaler.fit_transform([[weight]])[0][0]
    bmi_scaled = scaler.fit_transform([[bmi]])[0][0]

    #predict clusters
    steps_cluster = daily_model.predict([[steps_scaled, distance_scaled]])
    calories_cluster = calories_model.predict([[calories_scaled]])
    activity_cluster = steps_model.predict([[sedentary_scaled, lightly_active_scaled, fairly_active_scaled, very_active_scaled]])
    heart_rate_cluster = heart_rate_model.predict([[heart_rate_scaled]])
    sleep_cluster = sleep_model.predict([[sleep_hours_scaled]])
    weight_cluster = weight_model.predict([[weight_scaled, bmi_scaled]])

    # Prepare context for the LLM

    user_context = f"""
    User Fitness Details:
    Steps: {steps}, Distance: {distance} km, Calories: {calories}
    Sedentary: {sedentary} min, Lightly Active: {lightly_active} min, Fairly Active: {fairly_active} min, Very Active: {very_active} min
    Heart Rate: {heart_rate} bpm, Sleep: {sleep_hours} min, Weight: {weight} kg, BMI: {bmi}

    Cluster Assignments:
    Steps Cluster: {steps_cluster[0]}, Calories Cluster: {calories_cluster[0]}, Activity Cluster: {activity_cluster[0]}
    Heart Rate Cluster: {heart_rate_cluster[0]}, Sleep Cluster: {sleep_cluster[0]}, Weight/BMI Cluster: {weight_cluster[0]}
    """

    prompt = f"""Given the following user fitness details and their cluster assignments, generate a personalized, structured diet plan for the user. 
    The plan should include breakfast, lunch, dinner, snacks, and daily calorie/macronutrient targets. 
    Also, be ready to answer follow-up questions about the plan.

    {user_context}
    """
    diet_plan = model.invoke(prompt)
    st.session_state['diet_plan'] = diet_plan
    st.session_state['chat_history'] = [
        {"role": "assistant", "content": f"Here is your personalized diet plan:\n{diet_plan}"}
    ]

# --- Chat Interface ---
if 'diet_plan' in st.session_state:
    st.header("💬 Ask NutriBot about your Diet Plan or Fitness Details")

    for msg in st.session_state.get('chat_history', []):
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    user_msg = st.chat_input("Ask a question about your diet plan or fitness...")
    if user_msg:
        st.session_state['chat_history'].append({"role": "user", "content": user_msg})

        # Compose context for LLM
        context = f"Diet Plan: {st.session_state['diet_plan']}\nUser Question: {user_msg}"
        response = model.invoke(context)
        st.session_state['chat_history'].append({"role": "assistant", "content": response})

        st.rerun()

