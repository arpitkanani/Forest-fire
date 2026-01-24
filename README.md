# 🔥 Algerian Forest Fire Weather Index Prediction

### Deployed on Render Link :https://fire-whether-index-prediction.onrender.com/
## 📌 Overview
This project focuses on predicting the **Fire Weather Index (FWI)** using
**regression-based machine learning models**.
FWI is a continuous indicator that represents the **potential intensity and severity of forest fires**.

The model is trained on the **Algerian Forest Fires dataset**, which contains
meteorological and fire-related measurements from two regions of Algeria.

---

## 🎯 Problem Statement
Forest fires cause significant environmental and economic damage.
Accurately estimating fire severity in advance is crucial for prevention and disaster management.

**Goal:**  
Build a machine learning regression model that predicts the **Fire Weather Index (FWI)**
based on weather conditions, fire indicators, and regional information.

---

## 🌍 Dataset Description
The dataset consists of daily observations collected during the **high fire-risk season (June–September)**.

### 🔹 Regions Covered
- **Bejaia Region (Region = 0)**  
- **Sidi-Bel-Abbes Region (Region = 1)**  

---

## 📊 Input Features
- **Temperature** – Ambient temperature (°C)  
- **RH** – Relative Humidity (%)  
- **Ws** – Wind speed (km/h)  
- **Rain** – Rainfall amount (mm)  
- **FFMC** – Fine Fuel Moisture Code  
- **DMC** – Duff Moisture Code  
- **ISI** – Initial Spread Index  
- **Classes** – Fire occurrence indicator  
  - `0` → No Fire  
  - `1` → Fire  
- **Region** – Geographical region  
  - `0` → Bejaia  
  - `1` → Sidi-Bel-Abbes  

### 🎯 Target Variable
- **FWI (Fire Weather Index)** – Continuous value indicating fire severity

---

## 🧠 Machine Learning Approach
- **Problem Type:** Regression  
- **Pipeline Steps:**
  - Data cleaning and preprocessing
  - Feature scaling
  - Handling regional and class indicators
  - Training multiple regression models
  - Selecting the best-performing model

---

## 📈 Model Evaluation
The regression model is evaluated using:
- **R² Score**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**

---

## 🔬 Experiment Tracking
- **MLflow** is used for:
  - Logging model parameters
  - Tracking evaluation metrics
  - Saving trained models
- Experiments are tracked using **DAGsHub MLflow UI**

---

## 🌐 Web Application
A **Flask-based web application** allows users to:
- Enter meteorological and fire-related inputs
- Predict **Fire Weather Index (FWI)** instantly
- View results in a responsive UI

---

## 🚀 Deployment
- Deployed on **Render**
- Production-ready setup using **Gunicorn**
- Linux-compatible dependencies

---

## 🛠 Tech Stack
- **Language:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn  
- **Experiment Tracking:** MLflow, DAGsHub  
- **Web Framework:** Flask  
- **Deployment:** Render  
- **Version Control:** Git, GitHub  

---

## 📂 Project Structure
Algerian-Forest-Fire-Prediction/
│
├── src/
│ ├── components/
│ ├── pipelines/
│ └── utils/
│
├── templates/
│ ├── index.html
│ └── home.html
│
├── app.py
├── requirements.txt
├── README.md
└── artifacts/


---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Algerian-Forest-Fire-Prediction.git
cd Algerian-Forest-Fire-Prediction

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt

python app.py

run above line in step by step on CMD

open in browser -- http://127.0.0.1:5000

