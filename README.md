# 🚗 Car Price Prediction System

An end-to-end Machine Learning application that predicts market prices for vehicles based on key technical specifications.

---

### 🚀 [Click Here to Launch the Live Web App](https://carpriceprediction-njjrgteuz8zi4pz9emakz4.streamlit.app/)

---

## 📌 Project Overview
This project presents an interactive Streamlit web application powered by a Machine Learning model. Users can input specific vehicle attributes—such as horsepower, engine size, curb weight, highway MPG, fuel type, and body style—to obtain a market price valuation in real-time.

## 🛠️ Tech Stack
- **Language**: Python
- **Web Framework**: Streamlit
- **Machine Learning**: Scikit-Learn, Pandas, NumPy
- **Model Persistence**: Joblib

## 📂 Project Structure
├── app.py                   # Streamlit web application code
├── comp_Car_Price1.ipynb   # Notebook containing EDA and model training
├── model.pkl                # Trained machine learning model package
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
pip install -r requirements.txt
streamlit run app.py
