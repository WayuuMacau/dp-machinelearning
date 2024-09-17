import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import altair as alt

st.title('🐧Penguin Classifier App')
st.subheader('🤖 machine learning model - support vector machine')
st.info('Designed by Lawrence Ma 🇲🇴 +853 62824370 or 🇭🇰 +852 55767752')
st.warning("Try to fine-tune the left-hand side parameters to see the prediction result of penguin species")

with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    st.write(df)
    st.write('**X - Independent variables**')
    X_raw = df.drop('species', axis=1)
    st.write(X_raw)
    st.write('**y - Dependent variable**')
    y_raw = df.species
    st.write(y_raw)

    # Convert the target variable y to numeric (if it's categorical)
    y_numeric = pd.Series(y_raw.map({'Adelie': 1, 'Chinstrap': 2, 'Gentoo': 3}), name='species')

    # Combine X and y for correlation calculation
    combined_df = pd.concat([X_raw, y_numeric], axis=1)

# Input features
with st.sidebar:
    st.header('Input features')
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender', ('male', 'female'))
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))

    # Create a DataFrame for the input features
    data = {'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': gender,
            'island': island}
    input_df = pd.DataFrame(data, index=[0])

with st.expander('Data visualization'):
    colors = {
        'Adelie': 'rgb(0, 0, 255)',       # Blue
        'Chinstrap': 'rgb(255, 165, 0)',  # Orange
        'Gentoo': 'rgb(0, 128, 0)'        # Green
    }
    df['color'] = df['species'].map(colors)

    # 第一個散點圖
    scatter1 = alt.Chart(df).mark_circle(size=60).encode(
        x='bill_length_mm',
        y='body_mass_g',
        color=alt.Color('color:N', scale=None),
        tooltip=['species']
    ).interactive()

    # Add solid red spot
    red_spot1 = alt.Chart(input_df).mark_point(color='red', size=100, opacity=1).encode(
        x='bill_length_mm',
        y='body_mass_g'
    )

    st.altair_chart(scatter1 + red_spot1, use_container_width=True)

    # 第二個散點圖
    scatter2 = alt.Chart(df).mark_circle(size=60).encode(
        x='bill_depth_mm',
        y='flipper_length_mm',
        color=alt.Color('color:N', scale=None),
        tooltip=['species']
    ).interactive()

    # Add solid red spot
    red_spot2 = alt.Chart(input_df).mark_point(color='red', size=100, opacity=1).encode(
        x='bill_depth_mm',
        y='flipper_length_mm'
    )

    st.altair_chart(scatter2 + red_spot2, use_container_width=True)

    # 第三個散點圖
    scatter3 = alt.Chart(df).mark_circle(size=60).encode(
        x='bill_depth_mm',
        y='bill_length_mm',
        color=alt.Color('color:N', scale=None),
        tooltip=['species']
    ).interactive()

    # Add solid red spot
    red_spot3 = alt.Chart(input_df).mark_point(color='red', size=100, opacity=1).encode(
        x='bill_depth_mm',
        y='bill_length_mm'
    )

    st.altair_chart(scatter3 + red_spot3, use_container_width=True)

    # 第四個散點圖
    scatter4 = alt.Chart(df).mark_circle(size=60).encode(
        x='flipper_length_mm',
        y='body_mass_g',
        color=alt.Color('color:N', scale=None),
        tooltip=['species']
    ).interactive()

    # Add solid red spot
    red_spot4 = alt.Chart(input_df).mark_point(color='red', size=100, opacity=1).encode(
        x='flipper_length_mm',
        y='body_mass_g'
    )

    st.altair_chart(scatter4 + red_spot4, use_container_width=True)

# Correlation expander
with st.expander('Correlation'):
    # Ensure all data is numeric
    combined_df = combined_df.select_dtypes(include=[np.number])

    # Calculate correlation of each feature with the target variable
    correlation_with_y = combined_df.corr()['species'].drop('species')

    # Create a DataFrame for better display
    correlation_df = correlation_with_y.reset_index()
    correlation_df.columns = ['Feature', 'Correlation with y']

    # Display the correlation table with narrow columns
    st.write('**Correlation between each feature and the target variable**')
    st.dataframe(correlation_df, use_container_width=False)

# SVM Regression Metrics
with st.expander('Cross Validation'):
    st.caption('Train set 80%, Test set 20%; Sampling without replacement')
    # Prepare data for regression
    X_numeric = combined_df.drop('species', axis=1)
    y_numeric = combined_df['species']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_numeric, test_size=0.2, random_state=42)

    # Create and fit the SVM regressor
    svm_regressor = SVR(kernel='rbf')  # You can also try 'rbf' or other kernels
    svm_regressor.fit(X_train, y_train)

    # Make predictions
    y_train_pred = svm_regressor.predict(X_train)
    y_test_pred = svm_regressor.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_train, y_train_pred)

    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'R² Score', 'MAE'],
        'Train': [train_mse, train_r2, train_mae],
        'Test': [test_mse, test_r2, test_mae]
    })

    # Set the Metric column as the index
    metrics_df.set_index('Metric', inplace=True)

    # Display the metrics in a table
    st.write('**Regression Metrics**')
    st.dataframe(metrics_df, use_container_width=False)

# Model training and inference
X = X_numeric  # Ensure X is defined
y = y_numeric  # Ensure y is defined

clf = SVC(kernel='poly', probability=True) 
clf.fit(X, y)

# Apply model to make predictions
input_row = input_df.drop(columns=['sex', 'island'])  # Drop non-numeric columns
prediction_proba = clf.predict_proba(input_row)

# 獲取概率最高的物種的索引
predicted_index = np.argmax(prediction_proba)

# 使用該索引來獲取預測的物種
predicted_species = list({'Adelie': 1, 'Chinstrap': 2, 'Gentoo': 3}.keys())[predicted_index]

# 顯示預測的物種
st.success(f"Predicted Species: {predicted_species}")

# Display prediction probabilities
df_prediction_proba = pd.DataFrame(prediction_proba, columns={'Adelie': 1, 'Chinstrap': 2, 'Gentoo': 3}.keys())
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
             })  
