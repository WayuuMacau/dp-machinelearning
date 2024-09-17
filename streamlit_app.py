import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split  # Import train_test_split
import altair as alt

# The rest of your code remains unchanged...

# SVM Regression Metrics
with st.expander('Regression Metrics'):
    # Prepare data for regression
    X_numeric = combined_df.drop('species', axis=1)
    y_numeric = combined_df['species']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_numeric, test_size=0.2, random_state=42)

    # Create and fit the SVM regressor
    svm_regressor = SVR(kernel='linear')  # You can also try 'rbf' or other kernels
    svm_regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = svm_regressor.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Mean Squared Error', 'R² Score', 'Mean Absolute Error'],
        'Value': [mse, r2, mae]
    })

    # Display the metrics in a table
    st.write('**Regression Metrics**')
    st.dataframe(metrics_df, use_container_width=False)

# Input features
with st.sidebar:
    st.header('Input features')
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender', ('male', 'female'))
    
    # Create a DataFrame for the input features
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': gender}
    input_df = pd.DataFrame(data, index=[0])
    input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
    st.write('**Input penguin**')
    st.dataframe(input_df, use_container_width=False)  # Set to False for narrow width

st.header("", divider="rainbow")

# Data preparation
# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
def target_encode(val):
    return target_mapper[val]

y = y_raw.apply(target_encode)

# Model training and inference
clf = SVC(kernel='poly', probability=True) 
clf.fit(X, y)

# Apply model to make predictions
prediction_proba = clf.predict_proba(input_row)

# 獲取概率最高的物種的索引
predicted_index = np.argmax(prediction_proba)

# 使用該索引來獲取預測的物種
predicted_species = list(target_mapper.keys())[predicted_index]

# 顯示預測的物種
st.success(f"Predicted Species: {predicted_species}")

# Display prediction probabilities
df_prediction_proba = pd.DataFrame(prediction_proba, columns=target_mapper.keys())
df_prediction_proba_percentage = df_prediction_proba * 100
df_prediction_proba_percentage = df_prediction_proba_percentage.round(2)

# 使用漂亮的數據框顯示概率
st.dataframe(df_prediction_proba_percentage, 
             column_config={ 
               'Adelie': st.column_config.ProgressColumn( 
                 'Adelie (%)', 
                 format='%f', 
                 width='medium', 
                 min_value=0, 
                 max_value=100 
               ), 
               'Chinstrap': st.column_config.ProgressColumn( 
                 'Chinstrap (%)', 
                 format='%f', 
                 width='medium', 
                 min_value=0, 
                 max_value=100 
               ), 
               'Gentoo': st.column_config.ProgressColumn( 
                 'Gentoo (%)', 
                 format='%f', 
                 width='medium', 
                 min_value=0, 
                 max_value=100 
               )
             })  # Ensure this is properly closed
