import streamlit as st
import pandas as pd
import pickle
from datetime import date


# --- 1. MODEL LOADING ---
# Use st.cache_resource to load the model only once and store it in cache
@st.cache_resource
def load_model():
    with open('prophet_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


model = load_model()

# --- 2. UI SETUP & USER INPUTS ---
st.title('Rossmann Sales Forecast Dashboard 📈')

col1, col2, col3 = st.columns(3)

with col1:
    st.header("🗓️ Select Date & Promo")
    forecast_date = st.date_input("Select a date to forecast", value=date(2015, 8, 1))
    promo_toggle = st.checkbox("Is there a promotion?", value=True)

with col2:
    st.header("🏪 Store Details")
    store_type = st.selectbox("Select Store Type", options=['a', 'b', 'c', 'd'], index=2)
    assortment = st.selectbox("Select Assortment",
                              options=['a', 'b', 'c'])  # Note: Assortment was not a feature in our model

with col3:
    st.header("🏫 School Holiday")
    school_holiday_toggle = st.checkbox("Is it a school holiday?")

# --- 3. FORECASTING LOGIC ---
if st.button("🚀 Generate Forecast", use_container_width=True):

    # --- Prepare the input data for the model ---
    # Create a dataframe with the correct columns and user inputs
    input_df = pd.DataFrame({
        'ds': [forecast_date],
        'Promo': [1 if promo_toggle else 0],
        'SchoolHoliday': [1 if school_holiday_toggle else 0],
        'StoreType_a': [1 if store_type == 'a' else 0],
        'StoreType_b': [1 if store_type == 'b' else 0],
        'StoreType_c': [1 if store_type == 'c' else 0],
        'StoreType_d': [1 if store_type == 'd' else 0]
    })

    # --- Generate the main forecast ---
    forecast = model.predict(input_df)
    predicted_sales = forecast['yhat'].iloc[0]

    # --- Generate the promo recommendation ---
    # Create a second input dataframe for the no-promo scenario
    input_df_no_promo = input_df.copy()
    input_df_no_promo['Promo'] = 0

    # Get the forecast without a promotion
    forecast_no_promo = model.predict(input_df_no_promo)
    sales_no_promo = forecast_no_promo['yhat'].iloc[0]

    # Calculate the sales lift
    promo_lift_percentage = ((predicted_sales - sales_no_promo) / sales_no_promo) * 100

    recommendation = ""
    if promo_lift_percentage > 15:
        recommendation = "✅ Recommendation: A promotion is highly effective and recommended."
    else:
        recommendation = "⚠️ Recommendation: A promotion has a minor impact on sales."

    # --- 4. DISPLAY RESULTS ---
    st.success("Forecast generated successfully!")

    st.metric(label="Predicted Sales for Selected Date", value=f"€{predicted_sales:,.2f}")

    st.subheader("💡 Promotion Analysis")
    st.metric(label=f"Predicted Sales Lift from Promotion", value=f"{promo_lift_percentage:.2f}%")
    st.info(recommendation)

    # You can also display the raw forecast dataframe
    with st.expander("See Raw Forecast Data"):
        st.dataframe(forecast)