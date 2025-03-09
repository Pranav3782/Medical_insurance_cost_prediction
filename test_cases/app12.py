#Implemented recommendation plan
# added alcohol question
import warnings
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Title and description
st.title("Medical Insurance Cost Prediction ")

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
tab1, tab2, tab3, tab4 = st.tabs(["Predict Insurance Cost", "Project Details", "Insurance Benefits", "Insurance Plans"])
with tab1:
    st.header("Predict Insurance Cost")

    # Collecting input from the user
    age = st.slider('Age', 18, 100, 30)
    bmi = st.number_input("Enter BMI:", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("Enter number of children:", min_value=0, max_value=10, value=1)
    smoker = st.selectbox("Are you a smoker?", ['yes', 'no'])
    alcohol = st.selectbox("Do you consume alcohol?", ['yes', 'no'])
    sex = st.selectbox("Enter gender", ['male', 'female'])

    # Get unique regions from the dataset for validation
    valid_regions = data['region'].unique()
    region = st.selectbox("Select region", valid_regions)

    # Define fixed price adjustments for alcohol consumption
    alcohol_yes_cost = 500  # Fixed cost if the user consumes alcohol
    alcohol_no_cost = 0   # Fixed cost if the user does not consume alcohol

    # Initialize predicted_cost outside of the if statement
    predicted_cost = None

    # When user clicks the button, predict the cost
    if st.button("Predict Insurance Cost"):
        # Create input_data in correct format for the model
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'sex': [sex],
            'region': [region]
        })

        # Perform the prediction
        predicted_cost = model.predict(input_data)

        # Add fixed alcohol cost based on user selection
        additional_cost = alcohol_yes_cost if alcohol == 'yes' else alcohol_no_cost
        total_cost = predicted_cost[0] + additional_cost  # Add fixed price to predicted cost

        # Display the predicted cost and total cost
       # st.success(f"Predicted Insurance Cost (Base): {predicted_cost[0]}")
        st.info(f"Predicted Insurance Cost: {total_cost}")

with tab2:
    st.header("Project Details")
    st.write("""
             
Project Overview: Insurance Cost Prediction Using Machine Learning
             
This project leverages machine learning techniques, specifically a Random Forest Regressor, 
to predict insurance costs based on user inputs such as age, BMI, number of children, smoking 
status, gender, and region. By using historical insurance data, the model can make 
personalized cost predictions for individuals, allowing them to understand their potential 
insurance costs without having to consult a human agent.
     
Real-Time Usefulness of the Project:
1. Instant Insurance Cost Prediction:
- Users can get an accurate prediction of their insurance costs in real-time by inputting 
their details into the app.
- This eliminates the waiting period for insurance agents to provide quotes.
             
2. Cost Comparison:
- After predicting insurance costs, the app compares the result with real-world insurance plans, 
allowing users to quickly find the best option.
- This saves time and effort in searching through different insurance companies and plans manually.
             
3. Financial Planning:
- The app can be used for long-term financial planning by predicting how insurance costs may change 
over time based on factors such as age and health risks.
- Users can adjust their savings or budgets accordingly to account for future healthcare needs.
             
How It Helps Avoid Reliance on Policy Agents ?
             
1.Cost Efficiency:
- Eliminates the need to pay brokerage fees or agent commissions. The app provides direct and 
cost-effective access to predictions and plan comparisons.
2. Self-Service Platform:
- Users can interact with the app at their convenience, making it easier to plan and research without
needing to schedule appointments with agents.
             
    """)

with tab3:
    st.header("Insurance Importance and Benefits")
    st.write("""
Insurance is a critical financial tool that provides peace of mind and financial security. Some key benefits include:
             
- Protection from unexpected financial burdens in case of medical emergencies.
- Safeguards your family and assets in case of unforeseen circumstances.
- Offers tax benefits in many countries, reducing your overall tax burden.
- Provides mental peace knowing you're prepared for life's uncertainties.
- The project uses historical data and machine learning algorithms to provide accurate insurance cost predictions, 
ensuring users receive information based on real-world data rather than subjective opinions.
- Users can see how their inputs directly affect the predicted insurance costs, promoting transparency in the prediction 
process. Clear explanations of the algorithms and methodologies used can also build trust.
             
What is Insurance ?
             
Insurance is a financial arrangement that provides protection against unexpected financial losses. In simple terms, 
it’s a safety net that helps you cover costs if something bad happens, like an accident, illness, or property damage. 
You pay a certain amount (called a premium) to the insurance company regularly, and in return, the company agrees to 
pay for specific financial losses if they occur.

Why is Insurance Useful for Everyone?
             
1. Financial Protection:
Insurance protects you from large, unexpected expenses. For example, if you have health insurance and need surgery, 
             the insurance will cover most of the cost, which would otherwise be very expensive.

2. Peace of Mind:
Knowing you are covered in case something happens helps reduce stress. Whether it's health problems, car accidents, 
or damage to your home, having insurance gives you peace of mind.
             
3. Income Protection:
Some types of insurance, like life insurance or disability insurance, provide support for your family if you are unable 
             to work or if you pass away. This ensures that your loved ones are financially protected.

4. Savings and Investment:
Certain insurance policies, such as life insurance, also act as savings plans or investments. Over time, these policies 
        grow and can provide a return on your investment.

When is the Right Time to Get Insurance?

- Early in Life:
             
Health Insurance: It is recommended to get health insurance when you are young and healthy. Premiums are usually lower at 
             a younger age, and this ensures that you’re covered before any health issues arise.

- Before a Major Life Event:
             
Getting Married: Marriage often brings shared financial responsibilities. Life insurance and health insurance for both 
             partners are crucial.

Having Children: This is an important time to get life and health insurance to protect your growing family.
             
Buying a House: Home insurance is essential to protect your property from damage or loss due to accidents, theft, or natural
              disasters.
             
As Soon as You Start Earning:
If you have an income, it’s the right time to start considering insurance. This ensures that if anything happens to you, your 
             financial obligations, such as loans or family expenses, are covered.

Insurance and How They Work ?
             
Health Insurance:
What it Covers: Medical expenses like doctor visits, hospital stays, and surgeries.
How it Works: You pay a monthly or yearly premium, and when you need medical care, the insurance covers most of the cost.

Who Needs It: Everyone, especially in countries where medical care is expensive. Health emergencies can happen unexpectedly, 
and medical bills can be huge without coverage.

    """)

with tab4:
    st.header("Compare Predicted Cost with Insurance Plans")
    
    if predicted_cost is not None:
        # Extract the first value of predicted_cost
        predicted_cost_value = predicted_cost[0]  # Extract the first element from the array

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
        features = data['features']  # Assuming this column exists in the dataset
        year = data['year']  # Assuming the year column exists in the dataset

        # Collect all matching plans based on the predicted cost
        matching_plans = []
        for i, price_range in enumerate(cost_prices):
            if price_range is not None:
                low, high = price_range
                if low <= predicted_cost_value <= high:
                    # If the predicted cost falls within the range, collect plan details
                    matching_plans.append({
                        "Insurer": insurer[i],
                        "Plan Name": plan_names[i],
                        "Coverage Amount": coverage_amount[i],
                        "Claim Settlement Ratio": claim_ratio[i],
                        "Year": year[i],
                        "Features": features[i]  # Include the features column
                    })

        # Display the results
        if matching_plans:
            st.write("The following plans match your predicted cost:")
            best_plan = None
            highest_claim_ratio = 0  # Initialize the variable to track the best claim ratio

            for plan in matching_plans:
                st.write(f"- **Insurer**: {plan['Insurer']}")
                st.write(f"- **Plan Name**: {plan['Plan Name']}")
                st.write(f"- **Coverage Amount**: {plan['Coverage Amount']}")
                st.write(f"- **Claim Settlement Ratio**: {plan['Claim Settlement Ratio']:.2f}")
                st.write(f"- **Year**: {plan['Year']}")
                st.write(f"- **Features**: {plan['Features']}")  # Display the features for each plan
                st.write("---")

                # Recommend the plan with the highest claim settlement ratio
                if plan['Claim Settlement Ratio'] > highest_claim_ratio:
                    highest_claim_ratio = plan['Claim Settlement Ratio']
                    best_plan = plan

            # Display the best plan recommendation
            if best_plan:
                st.subheader("Recommended Plan:")
                st.write(f"- **Insurer**: {best_plan['Insurer']}")
                st.write(f"- **Plan Name**: {best_plan['Plan Name']}")
                st.write(f"- **Coverage Amount**: {best_plan['Coverage Amount']}")
                st.write(f"- **Claim Settlement Ratio**: {best_plan['Claim Settlement Ratio']:.2f}")
                st.write(f"- **Year**: {best_plan['Year']}")
                st.write(f"- **Features**: {best_plan['Features']}")  # Display features of the recommended plan

                # Explanation for why this plan is preferred
                st.subheader("Why choose this plan?")
                st.write(f"This plan is recommended because it has the **highest claim settlement ratio ({best_plan['Claim Settlement Ratio']:.2f})**, meaning the insurer has a strong record of settling claims quickly and fairly. "
                         f"In addition, the coverage amount of **₹{best_plan['Coverage Amount']}** offers a significant level of financial protection. "
                         f"Moreover, the plan provides the following features: **{best_plan['Features']}**, which adds extra value and benefits compared to other options.")
        else:
            st.write("No matching plan found.")
    else:
        st.write("Please predict the insurance cost first to compare plans.")

