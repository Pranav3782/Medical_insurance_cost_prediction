#4 tabs 
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Title and description
st.title("Insurance Cost Prediction App")

# Load the dataset
data = pd.read_excel('test_set_with_prices.xlsx')

# Selecting the relevant columns for features and the target (total_price)
X = data[['age', 'bmi', 'children', 'smoker', 'sex', 'region']]
y = data['total_price']

# One-hot encode categorical columns: 'sex', 'smoker', 'region'
categorical_cols = ['sex', 'smoker', 'region']
numeric_cols = ['age', 'bmi', 'children']

# Define column transformer to apply transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Split the dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Predict Insurance Cost", "Project Details", "Insurance Benefits", "Suitable Insurance Plans"])

with tab1:
    st.header("Predict Insurance Cost")

    # Collecting input from the user
    age = st.number_input("Enter age:", min_value=0, max_value=120, value=30)
    bmi = st.number_input("Enter BMI:", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("Enter number of children:", min_value=0, max_value=10, value=1)
    smoker = st.selectbox("Are you a smoker?", ['yes', 'no'])
    sex = st.selectbox("Enter gender", ['male', 'female'])

    # Get unique regions from the dataset for validation
    valid_regions = data['region'].unique()
    region = st.selectbox("Select region", valid_regions)

    # When user clicks the button, predict the cost
    if st.button("Predict Insurance Cost"):
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'sex': [sex],
            'region': [region]
        })

        predicted_cost = model.predict(input_data)
        st.success(f"Predicted Insurance Cost: {predicted_cost[0]}")

with tab2:
    st.header("Project Details")
    st.write("""
    This project is an insurance cost prediction application built using machine learning.
    The app uses user-provided information such as age, BMI, number of children, smoking status, gender,
    and region to predict the overall insurance cost.
    The RandomForestRegressor algorithm is trained on a dataset with various insurance cost data.
    """)

with tab3:
    st.header("Insurance Importance and Benefits")
    st.write("""
    Insurance is a critical financial tool that provides peace of mind and financial security. Some key benefits include:
    - Protection from unexpected financial burdens in case of medical emergencies.
    - Safeguards your family and assets in case of unforeseen circumstances.
    - Offers tax benefits in many countries, reducing your overall tax burden.
    - Provides mental peace knowing you're prepared for life's uncertainties.
    """)

with tab4:
    st.header("Which Insurance Plan is Suitable?")
    st.write("""
    Choosing the right insurance plan depends on various factors such as age, health condition, financial situation, and personal needs. Here are some guidelines:
    - **Health Insurance:** Suitable for covering medical expenses.
    - **Life Insurance:** Ideal for protecting your family in case of unforeseen events.
    - **Car Insurance:** Necessary if you own a vehicle to cover potential damages or accidents.
    - **Home Insurance:** Recommended for homeowners to protect against property damage or theft.
    """)

# Light and Dark mode theme settings (removing side navigation)
st.markdown("""
    <style>
        /* Hide the dark mode switch */
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)
