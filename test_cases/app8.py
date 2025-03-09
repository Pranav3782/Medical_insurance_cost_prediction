#multiple predicted plans using the predicted cost details
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

#to supress the warnings occur  during the execution of the code
import warnings
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")


# Load the dataset
file_path = 'test_set_with_prices.xlsx'
data = pd.read_excel(file_path)

# Preprocess the data
data['smoker'] = data['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
data['sex'] = LabelEncoder().fit_transform(data['sex'])
data['region'] = LabelEncoder().fit_transform(data['region'])

# Tab layout
tab1, tab2 = st.tabs(["Insurance Cost Prediction", "Compare Plans"])

# Tab 1: Predict Insurance Cost
with tab1:
    st.header("Predict Insurance Cost")
    
    # Input fields
    age = st.slider('Age', 18, 100, 30)
    sex = st.selectbox('Sex', ['male', 'female'])
    bmi = st.number_input('BMI', 15.0, 50.0, 25.0)
    children = st.slider('Number of Children', 0, 5, 1)
    smoker = st.selectbox('Smoker', ['yes', 'no'])
    region = st.selectbox('Region', data['region'].unique())

    # Convert input values to numeric
    sex = 1 if sex == 'female' else 0
    smoker = 1 if smoker == 'yes' else 0

    # Define the input features
    input_data = [[age, sex, bmi, children, smoker, region]]

    # Train the model
    X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    y = data['total_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Predict the cost
    predicted_cost = model.predict(input_data)
    st.write(f"Predicted Insurance Cost: ₹{predicted_cost[0]:,.2f}")

# Tab 2: Compare Predicted Cost with Plans
# Tab 2: Compare Predicted Cost with Plans
with tab2:
    st.header("Compare Predicted Cost with Insurance Plans")
    
    # Helper function to parse the cost price range
    def parse_price_range(price):
        try:
            # Remove currency symbols and split into low and high range
            low, high = map(int, price.replace('₹', '').replace(',', '').split('-'))
            return (low, high)
        except Exception:
            return None  # In case of issues, return None
    
    cost_prices = data['cost_price'].apply(parse_price_range)
    insurer = data['insurer']
    plan_names = data['plan_name']
    coverage_amount = data['coverage_amount']
    claim_ratio = data['claim_settlement_ratio']
    year = data['year']  # Assuming the year column exists in the dataset

    # Collect all matching plans based on the predicted cost
    matching_plans = []
    for i, price_range in enumerate(cost_prices):
        if price_range is not None:
            low, high = price_range
            if low <= predicted_cost <= high:
                # If the predicted cost falls within the range, collect plan details
                matching_plans.append({
                    "Insurer": insurer[i],
                    "Plan Name": plan_names[i],
                    "Coverage Amount": coverage_amount[i],
                    "Claim Settlement Ratio": claim_ratio[i],
                    "Year": year[i]  # Add the year to the plan details
                })

    # Display the results
    if matching_plans:
        st.write("The following plans match your predicted cost:")
        for plan in matching_plans:
            st.write(f"- **Insurer**: {plan['Insurer']}")
            st.write(f"- **Plan Name**: {plan['Plan Name']}")
            st.write(f"- **Coverage Amount**: {plan['Coverage Amount']}")
            st.write(f"- **Claim Settlement Ratio**: {plan['Claim Settlement Ratio']:.2f}")
            st.write(f"- **Year**: {plan['Year']}")  # Display the year
            st.write("---")
    else:
        st.write("No matching plan found.")

