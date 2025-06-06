import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import r2_score  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import os

# Function to load data
@st.cache_data
def load_data(file_path="prediction_data.xlsx"):
    if not os.path.exists(file_path):
        st.error(f"The file '{file_path}' was not found. Please upload it to the working directory.")
        return None
    try:
        df = pd.read_excel(file_path, sheet_name="Sheet1")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to train Random Forest model
def train_rf_model(data, features, target):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    return model, score

# Function for Random Forest Future Prediction
def predict_rf_with_tariff(model, data, tariff_value, features, target, years_to_predict=3):
    last_year = data['Year'].max()
    future_years_list = [last_year + i for i in range(1, years_to_predict + 1)]
    mean_features = data[features].mean().to_dict()
    mean_features["Average Tariff Rate (%)"] = tariff_value

    future_data_df = pd.DataFrame()
    future_data_df['Year'] = future_years_list

    repeated_features_df = pd.DataFrame([mean_features] * years_to_predict, columns=features)
    predicted_values = model.predict(repeated_features_df)
    future_data_df[target] = predicted_values / 1e9  # Convert to billions
    return future_data_df

# Function for Linear Regression Prediction
def predict_lr_with_tariff(df, country, tariff_rate):
    X = df[['Year', 'Average Tariff Rate (%)']]
    y = df['Import Value (USD)']
    model = LinearRegression()
    model.fit(X, y)
    last_year = df['Year'].max()
    future_years_df = pd.DataFrame({
        'Year': [last_year + i for i in range(1, 4)],
        'Average Tariff Rate (%)': [tariff_rate] * 3
    })
    predictions = model.predict(future_years_df)
    return df['Year'], y, future_years_df['Year'], predictions, country

# --- Streamlit App Layout ---
st.title("India's Import Value Prediction")

# Load data
df = load_data()

if df is not None:
    features = ["Average Tariff Rate (%)", "GDP Growth (%)", "USD/INR (avg)", "Inflation (CPI, %)"]
    target = "Import Value (USD)"

    df_usa = df[df["Country"] == "USA"].copy()
    df_china = df[df["Country"] == "China"].copy()

    if not df_usa.empty and not df_china.empty:
        usa_model, usa_r2 = train_rf_model(df_usa, features, target)
        china_model, china_r2 = train_rf_model(df_china, features, target)

        st.metric(label="USA RF R² Score", value=f"{usa_r2:.2f}")
        st.metric(label="China RF R² Score", value=f"{china_r2:.2f}")

        st.sidebar.header("Settings")
        tariff_value = st.sidebar.slider("Select Tariff Rate (%)", min_value=0.0, max_value=30.0, step=0.5, value=5.0)
        years_to_predict = st.sidebar.slider("Years to Predict (RF)", min_value=1, max_value=5, step=1, value=3)

        st.header("Random Forest Future Predictions")

        usa_preds_rf = predict_rf_with_tariff(usa_model, df_usa, tariff_value, features, target, years_to_predict)
        china_preds_rf = predict_rf_with_tariff(china_model, df_china, tariff_value, features, target, years_to_predict)

        st.subheader("India Import from USA Random Forest Prediction")
        fig_usa_rf, ax_usa_rf = plt.subplots()
        ax_usa_rf.plot(usa_preds_rf['Year'], usa_preds_rf[target], marker='o', color='blue')
        ax_usa_rf.set_title(f"USA Import Value Predictions (Tariff: {tariff_value}%)")
        ax_usa_rf.set_xlabel("Year")
        ax_usa_rf.set_ylabel("Import Value (Billion USD)")
        ax_usa_rf.grid(True)
        st.pyplot(fig_usa_rf)
        st.dataframe(usa_preds_rf.rename(columns={target: 'Import Value (Billion USD)'}))

        st.subheader("India Import from China Random Forest Prediction")
        fig_china_rf, ax_china_rf = plt.subplots()
        ax_china_rf.plot(china_preds_rf['Year'], china_preds_rf[target], marker='o', color='red')
        ax_china_rf.set_title(f"China Import Value Predictions (Tariff: {tariff_value}%)")
        ax_china_rf.set_xlabel("Year")
        ax_china_rf.set_ylabel("Import Value (Billion USD)")
        ax_china_rf.grid(True)
        st.pyplot(fig_china_rf)
        st.dataframe(china_preds_rf.rename(columns={target: 'Import Value (Billion USD)'}))

        st.header("Linear Regression Forecasts (Next 3 Years)")

        us_data_lr = df_usa[['Year', 'Average Tariff Rate (%)', 'Import Value (USD)']]
        china_data_lr = df_china[['Year', 'Average Tariff Rate (%)', 'Import Value (USD)']]

        us_year_actual_lr, us_y_actual_lr, us_year_predicted_lr, us_predictions_lr, us_country_lr = predict_lr_with_tariff(us_data_lr, 'USA', tariff_value)

        st.subheader(f"{us_country_lr} Linear Regression Forecast")
        fig_usa_lr, ax_usa_lr = plt.subplots()
        ax_usa_lr.plot(us_year_actual_lr, us_y_actual_lr / 1e9, label='Actual')
        ax_usa_lr.plot(us_year_predicted_lr, us_predictions_lr / 1e9, label='Predicted', linestyle='--', marker='o')
        ax_usa_lr.set_title(f'{us_country_lr} - LR Forecast with {tariff_value}% Tariff Rate')
        ax_usa_lr.set_xlabel('Year')
        ax_usa_lr.set_ylabel('Import Value (Billion USD)')
        ax_usa_lr.legend()
        ax_usa_lr.grid(True)
        st.pyplot(fig_usa_lr)
        st.dataframe(pd.DataFrame({'Year': us_year_predicted_lr, 'Predicted Import Value (Billion USD)': us_predictions_lr / 1e9}))

        china_year_actual_lr, china_y_actual_lr, china_year_predicted_lr, china_predictions_lr, china_country_lr = predict_lr_with_tariff(china_data_lr, 'China', tariff_value)

        st.subheader(f"{china_country_lr} Linear Regression Forecast")
        fig_china_lr, ax_china_lr = plt.subplots()
        ax_china_lr.plot(china_year_actual_lr, china_y_actual_lr / 1e9, label='Actual')
        ax_china_lr.plot(china_year_predicted_lr, china_predictions_lr / 1e9, label='Predicted', linestyle='--', marker='o')
        ax_china_lr.set_title(f'{china_country_lr} - LR Forecast with {tariff_value}% Tariff Rate')
        ax_china_lr.set_xlabel('Year')
        ax_china_lr.set_ylabel('Import Value (Billion USD)')
        ax_china_lr.legend()
        ax_china_lr.grid(True)
        st.pyplot(fig_china_lr)
        st.dataframe(pd.DataFrame({'Year': china_year_predicted_lr, 'Predicted Import Value (Billion USD)': china_predictions_lr / 1e9}))
    else:
        st.warning("Data for USA or China not found in the file.")
