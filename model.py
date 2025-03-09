# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

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
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")

# Define a function to predict the insurance cost based on user input
def predict_insurance_cost():
    # Collecting input from the user
    age = int(input("Enter age: "))
    bmi = float(input("Enter BMI: "))
    children = int(input("Enter number of children: "))
    smoker = input("Are you a smoker? (yes/no): ").lower().strip()
    sex = input("Enter gender (male/female): ").lower().strip()

    # Get unique regions from the dataset for validation
    valid_regions = [region.strip().lower() for region in data['region'].unique()]  # Normalize dataset regions
    print(f"Valid regions: {valid_regions}")
    region = input("Enter region: ").lower().strip()  # Normalize user input

    # Debugging: Print the region input for verification
    print(f"User input for region: '{region}'")

    # Check if the input region is valid
    if region not in valid_regions:
        print(f"Invalid region. Please choose from the valid regions: {valid_regions}")
        return None  # Return None or handle this case as needed

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
    predicted_cost = model.predict(input_data)

    return predicted_cost[0]

# Call the function and print the predicted insurance cost
predicted_price = predict_insurance_cost()
if predicted_price is not None:
    print(f"Predicted Insurance Cost: {predicted_price}")
