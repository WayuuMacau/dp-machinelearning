import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import altair as alt
from supabase import create_client
from sklearn.preprocessing import LabelEncoder

st.title('🏠 Properties Price Predictor')
st.subheader('🤖 Machine Learning Model - Random Forest Regression')
st.info('Designed by Lawrence Ma 🇲🇴 +853 62824370 or 🇭🇰 +852 55767752')
st.warning("Try to fine-tune the left-hand side parameters to see the prediction result of property price")

# Supabase connection
url = "https://cbtanfncszzrrdebqxwp.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNidGFuZm5jc3p6cnJkZWJxeHdwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjgzOTMzNjUsImV4cCI6MjA0Mzk2OTM2NX0.oXLdobwdPfYDVImtFBj5Ubef5PMYdGpqSMcsyv0rYus"
supabase = create_client(url, key)

# Load data from Supabase
response = supabase.table('properties').select("*").execute()
df = pd.DataFrame(response.data)

X_raw = df.drop('price', axis=1)
y_raw = df.price

# Input features
with st.sidebar:
    st.header('Input features')
    bed = st.slider('Number of Bedrooms', int(X_raw['bed'].min()), int(X_raw['bed'].max()), 3)
    bath = st.slider('Number of Bathrooms', int(X_raw['bath'].min()), int(X_raw['bath'].max()), 2)
    acre_lot = st.slider('Total land size / lot size in acres', float(X_raw['acre_lot'].min()), float(X_raw['acre_lot'].max()), 0.5)
    city = st.selectbox('City', X_raw['city'].unique())
    house_size = st.slider('House size / living space in square feet', int(X_raw['house_size'].min()), int(X_raw['house_size'].max()), 500)

# Create a DataFrame for the input features
input_data = {
    'bed': [bed],
    'bath': [bath],
    'acre_lot': [acre_lot],
    'city': [city],
    'house_size': [house_size]
}
input_df = pd.DataFrame(input_data)

with st.expander('Data'):
    st.write('**Raw data**')
    st.write(df)
    st.write('**X - Independent variables**')
    st.write(X_raw)
    st.write('**y - Dependent variable**')
    st.write(y_raw)

with st.expander('Data visualization'):
    st.caption('Below coordinates of red circles are the parameters chosen by left sidebar.')

    # Bed vs Bath
    scatter1 = alt.Chart(df).mark_circle(size=60).encode(
        x='bed',
        y='bath',
        color='price:Q',
        tooltip=['bed', 'bath', 'price']
    ).interactive()

    red_circle1 = alt.Chart(input_df).mark_circle(size=100, color='red').encode(
        x='bed',
        y='bath'
    )

    st.altair_chart(scatter1 + red_circle1, use_container_width=True)

    # Bath vs Acre Lot
    scatter2 = alt.Chart(df).mark_circle(size=60).encode(
        x='bath',
        y='acre_lot',
        color='price:Q',
        tooltip=['bath', 'acre_lot', 'price']
    ).interactive()

    red_circle2 = alt.Chart(input_df).mark_circle(size=100, color='red').encode(
        x='bath',
        y='acre_lot'
    )

    st.altair_chart(scatter2 + red_circle2, use_container_width=True)

    # Acre Lot vs house_size
    scatter3 = alt.Chart(df).mark_circle(size=60).encode(
        x='acre_lot',
        y='house_size',
        color='price:Q',
        tooltip=['acre_lot', 'house_size', 'price']
    ).interactive()

    red_circle3 = alt.Chart(input_df).mark_circle(size=100, color='red').encode(
        x='acre_lot',
        y='house_size'
    )

    st.altair_chart(scatter3 + red_circle3, use_container_width=True)


# Correlation expander
with st.expander('Correlation & Feature Importances'):
    # 確保所有數據都是數字類型
    le = LabelEncoder()
    X_raw_encoded = X_raw.copy()
    X_raw_encoded['city'] = le.fit_transform(X_raw_encoded['city'])
    combined_df_numeric = pd.concat([X_raw_encoded, y_raw], axis=1)

    # 計算每個特徵與目標變量的相關性
    correlation_with_y = combined_df_numeric.corr()['price'].drop('price')

    # 創建 DataFrame 以便更好地顯示並按降序排序
    correlation_df = correlation_with_y.reset_index()
    correlation_df.columns = ['Feature', 'Correlation with y']
    correlation_df = correlation_df.sort_values('Correlation with y', ascending=False)  # 按降序排序

    # 顯示相關性表格，移除左側的索引
    st.write('**Correlation between each feature and the target variable**')
    st.dataframe(correlation_df.set_index('Feature'), use_container_width=False)

    # 準備回歸數據
    X = X_raw_encoded
    y = y_raw

    # 創建並擬合隨機森林回歸模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    # 顯示特徵重要性
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    })

    # 根據相關性表格的順序重新排列特徵重要性
    feature_importance['Correlation'] = feature_importance['Feature'].map(correlation_with_y)
    feature_importance = feature_importance.sort_values('Correlation', ascending=False)

    st.write("**Feature Importances**")
    st.dataframe(feature_importance.set_index('Feature')[['Importance']], use_container_width=False)
    #st.bar_chart(feature_importance.set_index('feature'))

# Random Forest Regressor Metrics
with st.expander('Random Forest Regressor Metrics'):
    st.caption('Train set 80%, Test set 20%; Sampling without replacement')
    
    # Prepare data for regression
    X = X_raw_encoded
    y = y_raw

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the Random Forest regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)

    # Make predictions
    y_train_pred = rf_regressor.predict(X_train)
    y_test_pred = rf_regressor.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Calculate Adjusted R²
    n_train = X_train.shape[0]
    p_train = X_train.shape[1]
    train_adj_r2 = 1 - (1 - train_r2) * (n_train - 1) / (n_train - p_train - 1)

    n_test = X_test.shape[0]
    p_test = X_test.shape[1]
    test_adj_r2 = 1 - (1 - test_r2) * (n_test - 1) / (n_test - p_test - 1)

    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'Adjusted R²'],
        'Train': [train_mse, train_rmse, train_mae, train_r2, train_adj_r2],
        'Test': [test_mse, test_rmse, test_mae, test_r2, test_adj_r2]
    })

    # Set the Metric column as the index
    metrics_df.set_index('Metric', inplace=True)

    # Display the metrics in a table
    st.write('**Random Forest Regressor Metrics**')
    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=False)

    # Add summary text
    st.markdown("""
    **Summary:**
    
    The Random Forest regressor shows good performance on both the training and test sets. 
    
    - MSE and RMSE: These metrics indicate the average squared and absolute differences between predicted and actual values.
    - MAE: This represents the average absolute difference between predicted and actual values.
    - R² and Adjusted R²: These metrics indicate how well the model explains the variance in the target variable.

    The close values between train and test metrics suggest that the model is generalizing well and not overfitting significantly. However, the slightly higher values for the training set indicate there might be a small amount of overfitting, which is common and often acceptable if not too large.
    """)
    
with st.expander('Input features'):
    st.caption('Below values are the parameters chosen by left sidebar.')
    st.dataframe(input_df, use_container_width=False)  # Set to False for narrow width

st.header("", divider="rainbow")

# Data preparation
# Encode input
input_df_encoded = input_df.copy()
input_df_encoded['city'] = le.transform(input_df_encoded['city'])

# Model training and inference
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Apply model to make predictions
prediction = rf_model.predict(input_df_encoded)

# Display the predicted price
st.success(f"Predicted Price: ${prediction[0]:,.2f}")


