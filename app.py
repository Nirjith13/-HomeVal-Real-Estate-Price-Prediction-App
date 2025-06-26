import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 House Price Prediction App")
st.write("A simple ML app to predict house prices based on various features")

# Sidebar
st.sidebar.header("Model Configuration")

# Generate sample data
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    size = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    age = np.random.randint(0, 50, n_samples)
    location_score = np.random.uniform(1, 10, n_samples)
    
    # Generate target with some realistic relationships
    price = (
        size * 150 +
        bedrooms * 10000 +
        bathrooms * 15000 +
        (50 - age) * 1000 +
        location_score * 20000 +
        np.random.normal(0, 30000, n_samples)
    )
    
    df = pd.DataFrame({
        'size_sqft': size,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': age,
        'location_score': location_score,
        'price': price
    })
    
    return df.round(2)

# Load data
df = generate_sample_data()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Dataset Overview")
    st.write(f"Dataset contains {len(df)} house records with the following features:")
    st.dataframe(df.head(10))
    
    # Basic statistics
    st.subheader("📈 Data Statistics")
    st.write(df.describe())

with col2:
    st.header("🎯 Make Prediction")
    
    # Input features for prediction
    st.subheader("Enter House Details:")
    
    size_input = st.number_input("Size (sq ft)", min_value=500, max_value=5000, value=2000)
    bedrooms_input = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5], index=2)
    bathrooms_input = st.selectbox("Bathrooms", options=[1, 2, 3, 4], index=1)
    age_input = st.slider("Age (years)", min_value=0, max_value=50, value=10)
    location_input = st.slider("Location Score (1-10)", min_value=1.0, max_value=10.0, value=7.0, step=0.1)

# Model selection
st.sidebar.subheader("Choose Model")
model_choice = st.sidebar.selectbox(
    "Select ML Algorithm",
    ["Linear Regression", "Random Forest"]
)

# Train model
X = df[['size_sqft', 'bedrooms', 'bathrooms', 'age_years', 'location_score']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model_choice == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model performance
st.header("📊 Model Performance")
col1, col2, col3 = st.columns(3)

with col1:
    mse = mean_squared_error(y_test, y_pred)
    st.metric("Mean Squared Error", f"{mse:,.0f}")

with col2:
    r2 = r2_score(y_test, y_pred)
    st.metric("R² Score", f"{r2:.3f}")

with col3:
    rmse = np.sqrt(mse)
    st.metric("Root MSE", f"{rmse:,.0f}")

# Prediction
st.header("🎯 Your House Price Prediction")

if st.button("Predict Price", type="primary"):
    input_data = np.array([[size_input, bedrooms_input, bathrooms_input, age_input, location_input]])
    predicted_price = model.predict(input_data)[0]
    
    st.success(f"Predicted House Price: ${predicted_price:,.2f}")
    
    # Show input summary
    st.write("**Input Summary:**")
    st.write(f"- Size: {size_input:,} sq ft")
    st.write(f"- Bedrooms: {bedrooms_input}")
    st.write(f"- Bathrooms: {bathrooms_input}")
    st.write(f"- Age: {age_input} years")
    st.write(f"- Location Score: {location_input}/10")

# Visualizations
st.header("📈 Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price vs Size")
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(df['size_sqft'], df['price'], alpha=0.6)
    plt.xlabel('Size (sq ft)')
    plt.ylabel('Price ($)')
    plt.title('House Price vs Size')
    st.pyplot(fig)

with col2:
    st.subheader("Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    st.pyplot(fig)

# Feature importance (for Random Forest)
if model_choice == "Random Forest":
    st.subheader("🎯 Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
    plt.title('Feature Importance')
    st.pyplot(fig)

# Footer
st.markdown("---")
