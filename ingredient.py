# Import necessary libraries
#from google.colab import sheets
#sheet = sheets.InteractiveSheet(df=food_data)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load datasets
food_data = pd.read_csv('Food Ingredients and Recipe Dataset with Image Name Mapping.csv')
allergy_data = pd.read_csv('FoodData.csv')  # Replace with the allergy dataset file path

# Preprocessing: Clean and extract ingredients
food_data['Cleaned_Ingredients'] = food_data['Cleaned_Ingredients'].apply(eval)  # Convert stringified list to actual list

# Label encoding for allergies
le = LabelEncoder()
allergy_data['Allergy_Label'] = le.fit_transform(allergy_data['Allergy'])

# One-hot encoding for multi-class classification of ingredients
ohe = OneHotEncoder()
ingredient_encoded = ohe.fit_transform(allergy_data[['Food']])

# Train the model
X_train, X_test, y_train, y_test = train_test_split(ingredient_encoded, allergy_data['Allergy_Label'], test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict allergy risk based on user's lookup table
def predict_allergy_risk(user_lookup_table, detected_ingredients):
    risky_ingredients = set(user_lookup_table) & set(detected_ingredients)
    if risky_ingredients:
        return f"Avoid! Risky ingredients: {', '.join(risky_ingredients)}"
    else:
        return "Good to go!"

# Get user input for detected ingredients
detected_ingredients_input = input("Enter detected ingredients (comma-separated): ")

# Convert the input into a list of ingredients
detected_ingredients = [ingredient.strip() for ingredient in detected_ingredients_input.split(',')]

# Example usage: Replace with actual user allergy data
user_lookup_table = ['Pecorino', 'cheese']  # This represents user allergies
risk_message = predict_allergy_risk(user_lookup_table, detected_ingredients)
print(risk_message)