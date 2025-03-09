# tab1 is working fine but not able to display the plan using the cost 
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
file_path = r'test_set_with_prices.xlsx'
data = pd.read_excel(file_path)

# Streamlit app setup
st.title("Insurance Cost Predictor and Plan Recommender")

# Create tabs
tab1, tab2 = st.tabs(["Prediction", "Plan Details"])

# Preprocess data for linear regression model
# Selecting the relevant columns for the model
data = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'total_price']]

# Separate features and target variable
X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = data['total_price']

# Preprocessing: OneHotEncode categorical variables (sex, smoker, region)
categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']

# Create a ColumnTransformer to apply transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Create a pipeline that first transforms the data and then applies Linear Regression
model = Pipeline(steps=[('preprocessor', preprocessor), 
                        ('regressor', LinearRegression())])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Tab 1: Prediction
with tab1:
    st.header("Insurance Cost Prediction")

    # Input fields
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", options=['male', 'female'])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    region = st.selectbox("Region", options=data['region'].unique())
    smoker = st.selectbox("Smoker", options=['yes', 'no'])
    children = st.slider("Children", min_value=0, max_value=5, value=1)

    # Create input data for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    # Predict the insurance cost using the trained model
    predicted_cost = model.predict(input_data)[0]
    st.write(f"The predicted insurance cost is: ₹{predicted_cost:.2f}")

# Tab 2: Plan Details
# Tab 2: Plan Details
with tab2:
    st.header("Recommended Insurance Plans")

    # Check if predicted cost is available
    if 'predicted_cost' in st.session_state:
        predicted_cost = st.session_state['predicted_cost']
        
        # Define a range for filtering plans (e.g., ±10% of predicted cost)
        lower_bound = predicted_cost * 0.9
        upper_bound = predicted_cost * 1.1

        # Filter plans based on the predicted cost range
        # Assuming 'total_price' is the cost column in the dataset
        plan_data = data[data['total_price'].between(lower_bound, upper_bound)]

        if not plan_data.empty:
            st.write("Here are some suitable plans based on your predicted cost:")
            for idx, row in plan_data.iterrows():
                insurer = row.get('Insurer', 'N/A')
                plan_name = row.get('Plan Name', 'N/A')
                coverage_amount = row.get('Coverage Amount', 'N/A')
                claim_ratio = row.get('Claim Settlement Ratio (%)', 'N/A')

                st.markdown(f"""
                **Insurer**: {insurer}
                **Plan Name**: {plan_name}
                **Coverage Amount**: {coverage_amount}
                **Claim Settlement Ratio**: {claim_ratio if claim_ratio == 'N/A' else f"{claim_ratio}%"}
                ---
                """)
        else:
            st.write("No plans found in the given cost range.")
    else:
        st.write("Please predict the cost in the **Prediction** tab first.")