import subprocess

# Install all necessary packages
subprocess.check_call(["pip", "install", 
    "pandas", 
    "numpy", 
    "scikit-learn", 
    "matplotlib", 
    "ipywidgets", 
    "ipython"
])
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from IPython.display import display
import os

# Check if the file exists to avoid FileNotFoundError
file_path = "prediction_data.xlsx"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file '{file_path}' was not found. Please upload it to the working directory.")

# Load dataset
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Define features and target
features = ["Average Tariff Rate (%)", "GDP Growth (%)", "USD/INR (avg)", "Inflation (CPI, %)"]
target = "Import Value (USD)"

# Split data by country
df_usa = df[df["Country"] == "USA"]
df_china = df[df["Country"] == "China"]

# For Linear Regression based comparison
us_data = df_usa[['Year', 'Average Tariff Rate (%)', 'Import Value (USD)']]
china_data = df_china[['Year', 'Average Tariff Rate (%)', 'Import Value (USD)']]

# Train Random Forest Models
def train_model(data):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    return model, score

usa_model, usa_r2 = train_model(df_usa)
china_model, china_r2 = train_model(df_china)

print(f"USA Model R² Score: {usa_r2:.2f}")
print(f"China Model R² Score: {china_r2:.2f}")

# Random Forest Future Prediction Function
def predict_with_tariff(model, data, tariff_value, years_to_predict=3):
    last_year = data['Year'].max()
    future_years = [last_year + i for i in range(1, years_to_predict + 1)]
    mean_features = data[features].mean()
    mean_features["Average Tariff Rate (%)"] = tariff_value

    future_data = pd.DataFrame({"Year": future_years})
    repeated_features = pd.DataFrame([mean_features.values] * years_to_predict, columns=features)

    predicted_values = model.predict(repeated_features)
    future_data[target] = predicted_values
    return future_data

# Linear Regression Prediction Function
def predict_imports_interactive(df, country, future_tariff_rate):
    X = df[['Year', 'Average Tariff Rate (%)']]
    y = df['Import Value (USD)']
    model = LinearRegression()
    model.fit(X, y)
    future_years = pd.DataFrame({
        'Year': [2025, 2026, 2027],
        'Average Tariff Rate (%)': [future_tariff_rate]*3
    })
    predictions = model.predict(future_years)
    future_years['Predicted Import Value'] = predictions
    print(f"\n{country} Import Predictions (2025–2027) with {future_tariff_rate}% Tariff Rate:")
    print(future_years)
    return df['Year'], y, future_years['Year'], predictions, country

# Combined Interactive Plot Function
def update_plot(tariff):
    usa_preds = predict_with_tariff(usa_model, df_usa, tariff)
    china_preds = predict_with_tariff(china_model, df_china, tariff)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(usa_preds['Year'], usa_preds[target], marker='o', color='blue')
    plt.title(f"USA Import Value Predictions\n(Random Forest, Tariff Rate: {tariff}%)")
    plt.xlabel("Year")
    plt.ylabel("Import Value (USD)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(china_preds['Year'], china_preds[target], marker='o', color='red')
    plt.title(f"China Import Value Predictions\n(Random Forest, Tariff Rate: {tariff}%)")
    plt.xlabel("Year")
    plt.ylabel("Import Value (USD)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Linear Regression comparison
    us_year_actual, us_y_actual, us_year_predicted, us_predictions, us_country = predict_imports_interactive(us_data, 'USA', tariff)
    plt.figure(figsize=(8, 4))
    plt.plot(us_year_actual, us_y_actual, label='Actual')
    plt.plot(us_year_predicted, us_predictions, label='Predicted', linestyle='--', marker='o')
    plt.title(f'{us_country} - Linear Regression Forecast with {tariff}% Tariff Rate')
    plt.xlabel('Year')
    plt.ylabel('Import Value (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    china_year_actual, china_y_actual, china_year_predicted, china_predictions, china_country = predict_imports_interactive(china_data, 'China', tariff)
    plt.figure(figsize=(8, 4))
    plt.plot(china_year_actual, china_y_actual, label='Actual')
    plt.plot(china_year_predicted, china_predictions, label='Predicted', linestyle='--', marker='o')
    plt.title(f'{china_country} - Linear Regression Forecast with {tariff}% Tariff Rate')
    plt.xlabel('Year')
    plt.ylabel('Import Value (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Interactive Tariff Slider
interact(update_plot, tariff=FloatSlider(min=0.0, max=30.0, step=0.5, value=5.0, description='Tariff Rate (%)'))
