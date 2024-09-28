# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:27:47 2024

@author: Sai.Vigneshwar
"""

#pip install prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet


# Load the data (assuming we have a CSV or similar format)
data = pd.read_excel('C:/Users/Sai.Vigneshwar/OneDrive - Collaborate 365/Sai data (30-05-23)/Desktop/Sai/Python/Guvi/Dominos/Pizza_Sale.xlsx')

# Convert order_date to datetime format
data['order_date'] = pd.to_datetime(data['order_date'])

# Ensure order_time is in datetime format (if provided in string format)
data['order_time'] = pd.to_datetime(data['order_time'], format='%H:%M:%S').dt.time

# Check for missing values
missing_values = data.isnull().sum()

# Handle missing values for critical columns
critical_columns = ['pizza_name_id', 'total_price', 'pizza_category', 'pizza_name', 'pizza_ingredients']
#critical_columns = ['order_id', 'pizza_name_id', 'quantity', 'order_date', 'unit_price', 'pizza_ingredients']
# for col in critical_columns:
#     data = data.dropna(subset=[col])  # Drop rows with missing values in critical columns

# Ensure numerical columns are properly typed
data['quantity'] = pd.to_numeric(data['quantity'], errors='coerce')
data['unit_price'] = pd.to_numeric(data['unit_price'], errors='coerce')
data['total_price'] = pd.to_numeric(data['total_price'], errors='coerce')

# Create new features (day of week, month, hour from order_date and order_time)
data['day_of_week'] = data['order_date'].dt.dayofweek
data['month'] = data['order_date'].dt.month
data['hour'] = pd.to_datetime(data['order_time'], format='%H:%M:%S').dt.hour


#EDA

# Aggregate total pizzas sold over time
sales_over_time = data.groupby('order_date')['quantity'].sum().reset_index()

# Plot sales over time
plt.figure(figsize=(12, 6))
plt.plot(sales_over_time['order_date'], sales_over_time['quantity'], label='Total Pizzas Sold')
plt.xlabel('Order Date')
plt.ylabel('Total Pizzas Sold')
plt.title('Pizza Sales Over Time')
plt.legend()
plt.show()

# Sales by pizza category
category_sales = data.groupby('pizza_category')['quantity'].sum().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(x='pizza_category', y='quantity', data=category_sales)
plt.title('Pizza Sales by Category')
plt.xlabel('Pizza Category')
plt.ylabel('Total Pizzas Sold')
plt.show()

# Sales by pizza size
size_sales = data.groupby('pizza_size')['quantity'].sum().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(x='pizza_size', y='quantity', data=size_sales)
plt.title('Pizza Sales by Size')
plt.xlabel('Pizza Size')
plt.ylabel('Total Pizzas Sold')
plt.show()

# Sales by Day of the Week:
# Group by day of the week
weekly_sales = data.groupby('day_of_week')['quantity'].sum().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(x='day_of_week', y='quantity', data=weekly_sales)
plt.title('Pizza Sales by Day of the Week')
plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
plt.ylabel('Total Pizzas Sold')
plt.show()

# Split the pizza_ingredients column (assuming it's a comma-separated string)
data['ingredients'] = data['pizza_ingredients'].str.split(',')

# Flatten the ingredients list into individual rows and count them
ingredient_counts = data['ingredients'].explode().value_counts()

# Plot the top ingredients
plt.figure(figsize=(12, 6))
ingredient_counts.head(10).plot(kind='bar')
plt.title('Top Ingredients by Usage')
plt.xlabel('Ingredient')
plt.ylabel('Count')
plt.show()


#Feature Engineering

# Add time-based features
data['day_of_week'] = data['order_date'].dt.dayofweek
data['month'] = data['order_date'].dt.month
data['hour'] = pd.to_datetime(data['order_time'], format='%H:%M:%S').dt.hour

# Create binary holiday feature (this will depend on the actual holidays list)
holidays = ['2015-12-25', '2015-01-01', '2015-01-15', '2015-01-26', '2015-03-06', '2015-03-21', '2015-04-03', '2015-05-01', '2015-08-15', '2015-08-28', '2015-09-17', '2015-09-25', '2015-10-02', '2015-11-11']  # Example holidays
data['is_holiday'] = data['order_date'].isin(pd.to_datetime(holidays)).astype(int)


# Rolling averages
data['7_day_avg_sales'] = data['quantity'].rolling(window=7).mean().shift(1)  # 7-day moving average
data['30_day_avg_sales'] = data['quantity'].rolling(window=30).mean().shift(1)  # 30-day moving average

# Lag features: Sales from previous day, week, and two weeks ago
data['lag_1'] = data['quantity'].shift(1)  # Previous day's sales
data['lag_7'] = data['quantity'].shift(7)  # Sales from the previous week
data['lag_14'] = data['quantity'].shift(14)  # Sales from two weeks ago

# Drop rows with NaN values after creating lag and moving average features
# data = data.dropna()

#Model Selection
# Prepare data for Prophet (requires 'ds' as date and 'y' as the target variable)
prophet_data = data[['order_date', 'quantity']].rename(columns={'order_date': 'ds', 'quantity': 'y'})

# Initialize the model
model = Prophet()

# Add seasonality components if needed (e.g., weekly or monthly seasonality)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Train the model
model.fit(prophet_data)

# Make future predictions (let's forecast for the next 30 days)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Visualize the forecast
model.plot(forecast)
model.plot_components(forecast)


#Model valuation
# Filter forecast to match the historical period for evaluation
historical_forecast = forecast[forecast['ds'] <= data['order_date'].max()]

# Merge forecast with actual data to compute MAPE
merged_data = pd.merge(prophet_data, historical_forecast[['ds', 'yhat']], on='ds')

# Calculate MAPE
merged_data['error'] = (merged_data['y'] - merged_data['yhat']).abs() / merged_data['y']
mape = merged_data['error'].mean() * 100

print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


# Make future predictions for the next 7 days
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)

# Extract the predicted sales for the next week
predicted_sales = forecast[['ds', 'yhat']].tail(7)  # 'ds' is the date, 'yhat' is the forecasted quantity
print(predicted_sales)


# Assuming pizza_ingredients contains a list of ingredients for each pizza, e.g., "tomato, cheese, dough"
# Prepare a sample ingredient-to-pizza mapping for demonstration purposes
# Example data structure: {'pizza_name': ['ingredient1', 'ingredient2', 'ingredient3']}
# ingredient_mapping = {
#     'Margherita': ['tomato', 'cheese', 'dough'],
#     'Pepperoni': ['pepperoni', 'cheese', 'dough'],
#     'Veggie': ['bell_pepper', 'onion', 'tomato', 'cheese', 'dough'],
#     # Add more pizzas and their ingredients here
# }

ingredient_mapping = {
    'Chicken': ['Cheese', 'Provolone', 'Smoked', 'Gouda', 'Romano', 'Blue', 'Garlic', 'Tomatoes', 'Red', 'Peppers', 'Jalapeno', 'Onions', 'Cilantro', 'Corn', 'Sauce', 'Spinach', 'Mushrooms', 'Green', 'Olives', 'Feta', 'Artichokes', 'Pesto', 'Zucchini', 'Ricotta', 'Gorgonzola', 'Piccante', 'Parmigiano', 'Reggiano', 'Eggplant', 'Kalamata', 'Plum', 'Sun-dried'],
    'Classic': ['Anchovies', 'Bacon', 'Beef_Chuck_Roast' 'Capocollo', 'Chorizo_Sausage', 'Feta_Cheese', 'Garlic', 'Goat_Cheese', 'Green_Olives' 'Green_Peppers', 'Italian_Sausage', 'Kalamata_Olives', 'Mozzarella_Cheese', 'Mushrooms', 'Oregano', 'Pepperoni', 'Pineapple', 'Red_Onions', 'Red_Peppers', 'Sliced_Ham', 'Tomatoes'],
    'Supreme': ['duja Salami', 'Arugula', 'Artichokes', 'Asiago_Cheese', 'Brie_Carre_Cheese', 'Calabrese_Salami', 'Capocollo', 'Caramelized_Onions', 'Coarse_Sicilian_Salami', 'Fontina_Cheese', 'Friggitello_Peppers', 'Garlic', 'Genoa_Salami', 'Goat_Cheese', 'Green_Olives', 'Kalamata_Olives', 'Luganega_Sausage', 'Mozzarella_Cheese', 'Mushrooms', 'Onions', 'Pancetta', 'Pears', 'Pepperoni', 'Peperoncini_verdi', 'Prosciutto', 'Prosciutto_di_San_Daniele', 'Red_Onions', 'Soppressata_Salami', 'Spinach', 'Thyme', 'Tomatoes'],
    'Veggie': ['Artichokes', 'Blue_Cheese', 'Chipotle_Sauce', 'Cilantro', 'Corn', 'Feta_Cheese', 'Garlic', 'Gorgonzola_Piccante_Cheese', 'Green_Olives', 'Green_Peppers', 'Jalapeno_Peppers', 'Kalamata_Olives', 'Mozzarella_Cheese', 'Mushrooms', 'Parmigiano_Reggiano_Cheese', 'Pesto_Sauce', 'Plum_Tomatoes', 'Provolone_Cheese', 'Red_Onions', 'Red_Peppers', 'Ricotta_Cheese', 'Romano_Cheese', 'Smoked_Gouda_Cheese', 'Spinach', 'Sun-dried_Tomatoes', 'Tomatoes', 'Zucchini'],
    # Add more pizzas and their ingredients here
}

# Example sales forecast per pizza (dummy data, replace with actual forecast from model)
forecasted_pizza_sales = {
    'Chicken': 145,  # 100 pizzas predicted for Margherita
    'Classic': 150,   # 150 pizzas predicted for Pepperoni
    'Supreme': 80,       # 80 pizzas predicted for Veggie
    'Veggie': 125
}

# Ingredient aggregation: Calculate the required amount of each ingredient based on pizza sales
ingredient_demand = {}

# Loop through each pizza and its forecasted sales
for pizza, sales in forecasted_pizza_sales.items():
    # Get the list of ingredients for the pizza
    ingredients = ingredient_mapping[pizza]
    
    # For each ingredient in the pizza, add its demand based on forecasted sales
    for ingredient in ingredients:
        if ingredient in ingredient_demand:
            ingredient_demand[ingredient] += sales
        else:
            ingredient_demand[ingredient] = sales

# Print out the ingredient demand (total quantity required for each ingredient)
for ingredient, quantity in ingredient_demand.items():
    print(f"Ingredient: {ingredient}, Required Quantity: {quantity}")



# Create a purchase order from the ingredient demand
def generate_purchase_order(ingredient_demand):
    print("Purchase Order:")
    print("----------------------------")
    for ingredient, quantity in ingredient_demand.items():
        print(f"{ingredient}: {quantity} units required")

# Generate the purchase order
generate_purchase_order(ingredient_demand)