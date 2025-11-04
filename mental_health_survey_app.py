import streamlit as st    #importing required libraries
import torch
import pandas as pd
import joblib
import torch.nn as nn
st.set_page_config(page_title="Depression Prediction", page_icon="⚕️")  #streamlit app page title and the page icon

#mode and preprocessor loading
@st.cache_resource
def load_model():
    model = torch.load("/Users/sathya-22886/Downloads/MHS/mlp_depression_model.pth", map_location=torch.device("cpu"))
    return model

@st.cache_resource
def load_preprocessor():
    return joblib.load("/Users/sathya-22886/Downloads/MHS/preprocessor.pkl")

model_state_dict = torch.load("/Users/sathya-22886/Downloads/MHS/mlp_depression_model.pth", map_location="cpu")

# building the same model architecture
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[128, 64], dropout_p=0.3):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev_dim, h), 
                       nn.ReLU(),
                       nn.BatchNorm1d(h),
                       nn.Dropout(dropout_p)]
            prev_dim = h
        layers += [nn.Linear(prev_dim, 1), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

preprocessor = load_preprocessor()

#initialising model with the right input size

# Create a realistic dummy row with correct data types
dummy_df = pd.DataFrame([{
    "id": 0,  # optional, if your preprocessor ignores it
    "Name": "John Doe",
    "Gender": "Male",
    "City": "Chennai",
    "Working Professional or Student": "Student",
    "Profession": "Student",
    "Degree": "Bachelor's",
    "Age": 25,
    "Sleep Duration": "6-8 hours",
    "Dietary Habits": "Healthy",
    "Work/Study Hours": 6,
    "CGPA": 7.5,
    "Study Satisfaction": 3,
    "Job Satisfaction": 3,
    "Academic Pressure": 3,
    "Work Pressure": 3,
    "Financial Stress": 3,
    "Family History of Mental Illness": "No",
    "Have you ever had suicidal thoughts ?": "No"
}])

# Ensure numeric columns are numeric
numeric_cols = [
    "Age", "Work/Study Hours", "CGPA", "Study Satisfaction",
    "Job Satisfaction", "Academic Pressure", "Work Pressure", "Financial Stress"
]
dummy_df[numeric_cols] = dummy_df[numeric_cols].apply(pd.to_numeric, errors="coerce")

#Compute input dimension
input_dim = preprocessor.transform(dummy_df).shape[1]

model = MLP(input_dim)
model_state_dict = load_model()
model.load_state_dict(model_state_dict)
model.eval()

#streamlit UI
st.title("⚕️ Depression Prediction App")
st.markdown("Enter your details inorder to check the likelihood of depression.")

#user input features
age = st.number_input("Age", min_value=10, max_value=80, value=25)
sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "6-8 hours", "7-8 hours", "More than 8 hours"])
diet = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy"])
academic_pressure = st.slider("Academic Pressure (1–5)", 1, 5, 3)
work_pressure = st.slider("Work Pressure (1–5)", 1, 5, 3)
financial_stress = st.slider("Financial Stress (1–5)", 1, 5, 3)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
profession = st.text_input("Profession", "Student")

#dataframe for the user input
input_dict = {
    "Age": age,
    "Sleep Duration": sleep_duration,
    "Dietary Habits": diet,
    "Academic Pressure": academic_pressure,
    "Work Pressure": work_pressure,
    "Financial Stress": financial_stress,
    "Gender": gender,
    "Profession": profession,
}

user_df = pd.DataFrame([input_dict])

#Prediction button and result display

if st.button("Predict Depression Risk"):
    try:
        expected_cols = preprocessor.feature_names_in_   #adding column expected by preprocessor

        for col in expected_cols:
            if col not in user_df.columns:
                # Assign default values based on column type
                if col in ["Age", "Work/Study Hours", "CGPA", "Study Satisfaction",
                           "Job Satisfaction", "Academic Pressure", "Work Pressure", "Financial Stress"]:
                    user_df[col] = 0
                else:
                    user_df[col] = "Unknown"

        user_df = user_df[expected_cols]        #Ensure same column order as training data

        X_processed = preprocessor.transform(user_df)     #Preprocess and convert to tensor
        X_tensor = torch.tensor(X_processed, dtype=torch.float32)

        with torch.no_grad():
            prob = model(X_tensor).item()

     #Display result
        st.subheader("🧾 Prediction Result:")
        if prob >= 0.5:
            st.error(f"High Risk of Depression (Probability: {prob:.2f})")
        else:
            st.success(f"Low Risk of Depression (Probability: {prob:.2f})")

    except Exception as e:
        st.error(f"Error processing input: {e}")
