#  Rossmann Sales Forecasting Dashboard 📈

An interactive web application that forecasts daily sales for Rossmann stores and provides data-driven promotional recommendations. This project demonstrates a full data science workflow from data cleaning and modeling to deployment.

The `train.csv` is not available but you can find it on kaggle 

## Overview

This project uses historical sales data from over 1,000 Rossmann stores to train a time-series forecasting model. The final product is an interactive dashboard built with Streamlit where users can select a future date, store details, and promotional status to receive an instant sales prediction.

## Features

- **Interactive Forecasting:** Select any future date to get a sales prediction.
- **Dynamic Inputs:** Toggle inputs for promotions, school holidays, and store type.
- **Prescriptive Analytics:** The app calculates the potential sales lift from a promotion and provides a clear recommendation.
- **Data Visualization:** Displays the forecast trend and seasonal components.

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, Prophet, Scikit-learn, Streamlit, Pickle
- **Deployment:** Streamlit Community Cloud (or your choice)

## Project Structure
```
├── analysis.ipynb   # Saved, trained Prophet model
├── app.py              # The Streamlit dashboard script
├──     # Required Python packages
├── rossmann-store-sales/               # Folder for data files
│   ├── sample_submission.csv
│   └── store.csv
|   └── test.csv
└── README.md
```
## Setup and Usage

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-link]
    cd [your-repo-folder]
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Modeling Process

The predictive model was built using Facebook Prophet. The process involved:
1.  **Data Cleaning:** Handled missing values for competition data and merged store and sales datasets.
2.  **Feature Engineering:** One-hot encoded categorical features like `StoreType` and created a custom holiday calendar.
3.  **Model Training:** Trained the Prophet model using sales data and included `Promo`, `SchoolHoliday`, and `StoreType` as external regressors.
4.  **Hyperparameter Tuning:** A systematic grid search was performed to find the optimal parameters for `changepoint_prior_scale`, `seasonality_prior_scale`, and `holidays_prior_scale`, which reduced the final **RMSE to 1005.35**, a **33% improvement** over the baseline model.
