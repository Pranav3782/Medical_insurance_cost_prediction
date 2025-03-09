# Basic streamlit code 
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

# Streamlit application layout
st.title("Insurance Cost Prediction")

# Display the Mean Absolute Error (MAE)
#st.write(f"Model Mean Absolute Error: {mae}")

# User input fields
st.header("Enter your details")

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
