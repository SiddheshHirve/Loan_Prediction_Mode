🏦 Loan Approval Prediction App
This is an interactive Streamlit web application that predicts whether a customer is likely to be approved for a personal loan based on their financial profile and banking history.

The model is trained using a Random Forest Classifier, and the app takes user inputs such as age, income, credit card spending, and other financial indicators. The data is scaled using a pre-trained StandardScaler, and predictions are made on the fly.

✨ Features:
Predicts personal loan approval using real-time inputs

Displays prediction probability for transparency

Uses a trained machine learning model (RandomForestClassifier)

Scaled inputs with a saved StandardScaler object

Clean and responsive UI with Streamlit

Contact links included for GitHub, LinkedIn, and email

📁 Files:
loan_model.pkl: Trained machine learning model

scaler.pkl: Scaler used for feature normalization

app.py: Streamlit application script

🔧 Technologies Used:
Python

Scikit-learn

NumPy, Pandas

Streamlit for frontend
