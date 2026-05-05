import streamlit as st
import numpy as np
import pickle
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #020617;
    background-image: url("https://www.transparenttextures.com/patterns/cubes.png");
    color: #e2e8f0;
}

/* Cute Doodles */
body::before {
    content: "💳 💰 📊 🏦 💵 💸";
    position: fixed;
    top: 8%;
    left: 5%;
    font-size: 42px;
    opacity: 0.07;
    transform: rotate(-15deg);
}

body::after {
    content: "📈 💲 🏧 💼 💳";
    position: fixed;
    bottom: 8%;
    right: 5%;
    font-size: 42px;
    opacity: 0.07;
    transform: rotate(15deg);
}

/* Title */
h1 {
    text-align: center;
    color: #38bdf8;
    text-shadow: 0 0 18px #38bdf8;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background: rgba(15, 23, 42, 0.75);
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.15);
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

/* Inputs */
.stNumberInput input, .stSelectbox div {
    border-radius: 10px !important;
}

            /* Fix input labels (Age, Sex, etc.) */
label {
    color: #e2e8f0 !important;
    font-weight: 500;
}
/* Button */
.stButton>button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    color: black;
    font-weight: bold;
    font-size: 16px;
    transition: 0.3s;
}

.stButton>button:hover {
    box-shadow: 0 0 15px #38bdf8;
    transform: scale(1.03);
}

/* Result Box */
.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    margin-top: 20px;
}

/* Good */
.good {
    background: rgba(34, 197, 94, 0.1);
    color: #22c55e;
    text-shadow: 0 0 10px #22c55e;
}

/* Bad */
.bad {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    text-shadow: 0 0 10px #ef4444;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open('credit_risk_model.pkl', 'rb'))
encoders = joblib.load('encoders.pkl')

def encode_cols(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    return -1

# ---------------- HEADER ----------------
st.markdown("<h1>💳 Credit Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A smart system to evaluate loan risk instantly</p>", unsafe_allow_html=True)

# ---------------- PERSONAL INFO ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("👤 Personal Information")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 30)
with col2:
    sex = st.selectbox("Sex", ["male", "female"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FINANCIAL INFO ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("💼 Financial Information")

job_labels = {
    "Unskilled and Non-resident": 0,
    "Unskilled and Resident": 1,
    "Skilled": 2,
    "Highly Skilled": 3
}
job_selection = st.selectbox("Job Level", list(job_labels.keys()))
job = job_labels[job_selection]
housing = st.selectbox("Housing", ["own", "rent", "free"])

col3, col4 = st.columns(2)
with col3:
    saving = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich", "unknown"])
with col4:
    checking = st.selectbox("Checking Account", ["little", "moderate", "rich", "quite rich", "unknown"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- LOAN INFO ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Loan Details")

credit_amount = st.number_input("Credit Amount (₨ PKR)", min_value=0, value=31000)
duration = st.number_input("Duration (months)", min_value=1, value=12)
purpose = st.selectbox("Purpose", ["car", "education", "furniture/equipment", "radio/TV", "repairs", "business", "other"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Credit Risk"):

    applicant_data = {
        "Age": age,
        "Sex": sex.lower(),
        "Job": job,
        "Housing": housing.lower(),
        "Saving accounts": saving.lower(),
        "Checking account": checking.lower(),
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose.lower()
    }

    for col in ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]:
        le = encoders[col]
        applicant_data[col] = encode_cols(le, applicant_data[col])

    features = [
        "Age", "Sex", "Job", "Housing", "Saving accounts",
        "Checking account", "Credit amount", "Duration", "Purpose"
    ]

    X = np.array([[applicant_data[col] for col in features]])
    prediction = model.predict(X)[0]

    if prediction == 1:
        st.markdown('<div class="result-box bad">❌ High Credit Risk (BAD)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box good">✅ Low Credit Risk (GOOD)</div>', unsafe_allow_html=True)