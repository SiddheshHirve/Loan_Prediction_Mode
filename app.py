import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

# Set page config
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")

# App title
st.title("🏦 Loan Approval Prediction App")
st.markdown("Predict whether a customer will be approved for a personal loan based on their financial profile.")

# Sidebar Contact Info
with st.sidebar:
    st.markdown("### 📬 Connect with Me")
    st.markdown("""
    - 💻 [GitHub](https://github.com/SiddheshHirve)  
    - 🔗 [LinkedIn](https://www.linkedin.com/in/siddhesh-hirve-b52071294/)
    """)

st.markdown("---")

# Input section
st.header("📝 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('📅 Age', min_value=18, max_value=100, value=30)
    income = st.number_input('💰 Annual Income (in $1000)', value=50.0)
    family = st.selectbox('👨‍👩‍👧‍👦 Family Size', [1, 2, 3, 4])
    education = st.selectbox('🎓 Education Level', ['Undergrad', 'Graduate', 'Advanced/Professional'])
    mortgage = st.number_input('🏠 Mortgage Amount', value=0.0)

with col2:
    ccavg = st.number_input('💳 Avg. Credit Card Spending (in $1000)', value=2.0)
    securities_account = st.selectbox('📈 Securities Account?', ['No', 'Yes'])
    cd_account = st.selectbox('💽 CD Account?', ['No', 'Yes'])
    online = st.selectbox('🌐 Online Banking User?', ['No', 'Yes'])
    credit_card = st.selectbox('🧾 Has Credit Card?', ['No', 'Yes'])

# Map categorical inputs
education_map = {'Undergrad': 1, 'Graduate': 2, 'Advanced/Professional': 3}
binary_map = {'No': 0, 'Yes': 1}

# Feature preparation
features = np.array([[ 
    age, 
    income, 
    family, 
    ccavg, 
    education_map[education], 
    mortgage, 
    binary_map[securities_account], 
    binary_map[cd_account], 
    binary_map[online], 
    binary_map[credit_card] 
]])

# Scale inputs
features_scaled = scaler.transform(features)

# Prediction
if st.button("🔍 Predict Loan Approval"):
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    display_prob = min(probability * 100, 99.0)
    st.markdown("---")
    if prediction == 1:
        st.success(f"✅ Loan Approved! (Probability: **{probability*100:.2f}%**)")
    else:
        st.error(f"❌ Loan Not Approved. (Probability: **{probability*100:.2f}%**)")



