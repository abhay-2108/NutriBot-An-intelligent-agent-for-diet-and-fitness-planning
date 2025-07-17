# NutriBot – An Intelligent Agent for Diet and Fitness Planning

## Overview
NutriBot is an intelligent web-based agent designed to provide personalized diet and fitness plans based on user activity, health, and lifestyle data. Leveraging feature engineering and machine learning clustering (KMeans), NutriBot analyzes your fitness metrics and generates tailored diet recommendations, as well as answers follow-up questions about your plan.

## Features
- **Personalized Diet Plans:** Get structured meal plans (breakfast, lunch, dinner, snacks) and daily calorie/macronutrient targets based on your fitness profile.
- **Fitness Analysis:** Input your daily steps, distance, calories, activity breakdown, heart rate, sleep, weight, and BMI.
- **Interactive Chat:** Ask NutriBot questions about your diet plan or fitness details and receive intelligent responses.
- **Data-Driven Insights:** Utilizes clustering on real fitness datasets to categorize user profiles for more accurate recommendations.

## Dataset
The project uses real-world fitness and health data from the `Fitabase Data 4.12.16-5.12.16` directory, which includes:
- Daily and hourly activity, calories, steps, and intensities
- Heart rate and sleep logs
- Weight and BMI logs

These CSV files are preprocessed and clustered using KMeans to create user profile models for personalized recommendations.

## Model Training
The project performs feature engineering on the provided fitness dataset and trains KMeans clustering models for different aspects of user health (steps, calories, activity, heart rate, sleep, weight/BMI). The trained models are saved in the `model/` directory for use in the application.

- Models are trained on:
  - Steps and distance
  - Calories
  - Activity breakdown (sedentary, lightly/fairly/very active minutes)
  - Heart rate
  - Sleep duration
  - Weight and BMI
- Feature scaling is applied before clustering.

## Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/) for interactive web UI
- **ML/AI:** scikit-learn (KMeans), joblib, and [LangChain](https://python.langchain.com/) with Ollama LLM for natural language diet planning
- **Data Processing:** pandas, numpy

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/abhay-2108/NutriBot-An-intelligent-agent-for-diet-and-fitness-planning.git
   cd NutriBot-An-intelligent-agent-for-diet-and-fitness-planning
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download/prepare the dataset:**
   Ensure the `Fitabase Data 4.12.16-5.12.16` directory is present with all CSV files.
4. **Run the app:**
   ```bash
   streamlit run fitness-diet.py
   ```

## Usage
- Enter your fitness details in the sidebar.
- Click **Generate Plan** to receive a personalized diet and fitness plan.
- Use the chat interface to ask follow-up questions about your plan or fitness.

## Credits
- Fitness data: [Fitabase](https://www.fitabase.com/)
- ML/AI: scikit-learn, LangChain, Ollama
- UI: Streamlit

## License
This project is for educational and research purposes. 