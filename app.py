import pandas as pd
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = joblib.load("decision_tree_model_t.pkl")

# Page config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# === Background & Styling with GitHub Image ===
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://raw.githubusercontent.com/Agrita792/Churn_Prediction/main/background.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: black;
        font-family: 'Segoe UI', sans-serif;
    }}
    .main-box {{
        background-color: rgba(255, 255, 255, 0.92);
        padding: 2rem 3rem;
        border-radius: 18px;
        box-shadow: 0px 6px 12px rgba(0,0,0,0.15);
        margin: 0 auto;
    }}
    .stButton > button {{
        background-color: #43A047;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6em 2.5em;
        margin-top: 20px;
        transition: 0.3s ease;
        border: none;
    }}
    .stButton > button:hover {{
        background-color: #2E7D32;
        transform: scale(1.05);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# === Top Header Banner ===
st.markdown(
    """
    <div style='
        background-color: #0D47A1;
        padding: 1.5rem;
        border-radius: 0 0 12px 12px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.25);
    '>
        <h1 style='color: white; margin: 0; font-size: 2.8rem;'>📉 Customer Churn Prediction</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# === Title Section ===
st.markdown('<div class="main-box">', unsafe_allow_html=True)
st.write("Fill in customer usage details below to check if they are likely to churn.")

st.subheader("📋 Customer Information")

# === Single Column Layout ===
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 35)
no_of_days_subscribed = st.number_input("No. of Days Subscribed", min_value=1)
multi_screen = st.selectbox("Multi Screen Subscription", ["yes", "no"])
mail_subscribed = st.selectbox("Mail Subscribed", ["yes", "no"])
weekly_mins_watched = st.number_input("Weekly Minutes Watched")
minimum_daily_mins = st.number_input("Minimum Daily Minutes")
maximum_daily_mins = st.number_input("Maximum Daily Minutes")
weekly_max_night_mins = st.number_input("Weekly Max Night Minutes")
videos_watched = st.number_input("Videos Watched")
maximum_days_inactive = st.number_input("Max Days Inactive")
customer_support_calls = st.number_input("Customer Support Calls")

# Data Preprocessing
gender = 1 if gender == "Male" else 0
multi_screen = 1 if multi_screen == "yes" else 0
mail_subscribed = 1 if mail_subscribed == "yes" else 0

features = np.array([[gender, age, no_of_days_subscribed, multi_screen, mail_subscribed,
                      weekly_mins_watched, minimum_daily_mins, maximum_daily_mins,
                      weekly_max_night_mins, videos_watched, maximum_days_inactive,
                      customer_support_calls]])

st.divider()

if st.button("🔍 Predict Churn"):
    prediction = model.predict(features)[0]
    st.balloons()
    if prediction == 1:
        st.error("❌ Prediction: Customer is *likely to churn*.")
    else:
        st.success("✅ Prediction: Customer is *not likely to churn*.")

    # === EXPLANATION CHART ===
    st.markdown("### 📊 Why this prediction?")

    feature_names = [
        "Gender", "Age", "Days Subscribed", "Multi Screen", "Mail Subscribed",
        "Weekly Mins Watched", "Min Daily Mins", "Max Daily Mins",
        "Weekly Max Night Mins", "Videos Watched", "Max Days Inactive",
        "Support Calls"
    ]

    # Simulated average values (replace with real data if available)
    avg_values = np.array([0.5, 40, 200, 0.5, 0.5, 500, 30, 120, 300, 25, 5, 1])

    # Create comparison DataFrame
    user_values = features.flatten()
    comp_df = pd.DataFrame({
        "Feature": feature_names,
        "This Customer": user_values,
        "Average": avg_values
    }).set_index("Feature")

    # Plot chart
    fig, ax = plt.subplots(figsize=(8, 6))
    comp_df.plot(kind="barh", ax=ax, color=["#5CDB5C", "#FF7043"])
    ax.set_title("Comparison with Average Customer")
    ax.set_xlabel("Value")
    st.pyplot(fig)

st.markdown("</div>", unsafe_allow_html=True)