from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load dataset
house = pd.read_csv('linear_regression_housing_5000.csv')

# Features and target
y = house[['price_usd']]
x = house[['size_sqft','bedrooms','age_years','distance_km','has_garage','crime_index','school_rating']]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train model
lr2 = LinearRegression()
lr2.fit(x_train, y_train)

# Evaluate model
y_pred = lr2.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print("RMSE:", rmse)
print("R² Score:", r2)

# User input for prediction
print("\nEnter house details to predict price:")

size_sqft   = float(input("Enter size in sqft: "))
bedrooms    = int(input("Enter number of bedrooms: "))
age_years   = int(input("Enter age of house in years: "))
distance_km = float(input("Enter distance from city center (km): "))
has_garage  = int(input("Garage? (1 = Yes, 0 = No): "))
crime_index = float(input("Enter crime index: "))
school_rating = float(input("Enter school rating (1-10): "))


new_house = pd.DataFrame([[size_sqft, bedrooms, age_years, distance_km, has_garage, crime_index, school_rating]],
                         columns=['size_sqft','bedrooms','age_years','distance_km','has_garage','crime_index','school_rating'])

predicted_price = lr2.predict(new_house)
print(f"\nPredicted House Price: ${predicted_price[0][0]:,.2f}")
