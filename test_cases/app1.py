# 2 tabs , 1 is insurance cost , 1 is project implementation
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import streamlit as st

# Load your dataset
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
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # Handle unknown categories
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

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Streamlit application layout with tabs
st.title("Insurance Cost Prediction App")

# Create tabs
tab1, tab2 = st.tabs(["Predict Insurance Cost", "Project Details"])

# Tab 1: Insurance Cost Prediction
with tab1:
    st.header("Enter your details")

    # User input fields
    age = st.number_input("Enter your age", min_value=0, max_value=100, value=30)
    bmi = st.number_input("Enter your BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("Enter number of children", min_value=0, max_value=10, value=1)
    smoker = st.selectbox("Are you a smoker?", options=["yes", "no"])
    sex = st.selectbox("Select your gender", options=["male", "female"])

    # Get unique regions from the dataset
    valid_regions = sorted(data['region'].unique())
    region = st.selectbox("Select your region", options=valid_regions)

    # Predict button
    if st.button("Predict Insurance Cost"):
        # Create a dataframe with the input values
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'sex': [sex],
            'region': [region]
        })

        # Predict the insurance cost using the trained model
        predicted_cost = model.predict(input_data)[0]

        # Display the result
        st.success(f"Predicted Insurance Cost: ₹{predicted_cost:.2f}")

# Tab 2: Project Details
with tab2:
    st.header("Project Overview")
    st.write("""
    ### About the Project
    This project aims to build a machine learning model that predicts the medical insurance cost based on various factors such as:
    
    - **Age**: The age of the individual.
    - **BMI**: Body Mass Index, a measure of body fat based on height and weight.
    - **Number of Children**: Number of dependents.
    - **Smoking Status**: Whether the individual is a smoker or not.
    - **Gender**: Male or female.
    - **Region**: The geographical region where the individual resides.
    
    ### Dataset Information
    The dataset used for this project includes demographic and health information along with the corresponding insurance costs for various individuals. The features include:
    
    - Age
    - BMI
    - Number of Children
    - Smoking Status (yes/no)
    - Gender (male/female)
    - Region (e.g., Mysuru, Bengaluru, Mumbai, Delhi)
    
    The goal of the project is to train a model that accurately predicts the insurance cost for a given set of inputs using a **Random Forest Regressor**. This model has been evaluated using the **Mean Absolute Error (MAE)** metric, which indicates how well the model performs on unseen data.
    
    ### Technologies Used
    - **Python**: For model development and data processing.
    - **Scikit-learn**: For machine learning algorithms and evaluation.
    - **Pandas**: For data manipulation.
    - **Streamlit**: For building the web application interface.
    
    ### Future Improvements
    - Adding more features such as pre-existing medical conditions.
    - Improving model accuracy by trying different machine learning algorithms.
    - Providing more detailed explanations for the predicted insurance costs.
    """)

# End of the app
