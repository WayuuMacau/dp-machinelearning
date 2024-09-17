import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
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

    # Convert the variable to numeric (if it's categorical)
    y_numeric = pd.Series(y_raw.map({'Adelie': 1, 'Chinstrap': 2, 'Gentoo': 3}), name='species')

    # Combine X and y for correlation calculation
    combined_df = pd.concat([X_raw, y_numeric], axis=1)

# Input features
with st.sidebar:
    st.header('Input features')
    st.write('**Bill length (mm)**')
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    st.write('**Bill depth (mm)**')
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    st.write('**Flipper length (mm)**')
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    st.write('**Body mass (g)**')
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    st.write('**Gender**')
    gender = st.selectbox('Gender', ('male', 'female'))
    st.write('**Island**')
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))

    # Create a DataFrame for the input features
    input_data = {
        'Bill Length (mm)': bill_length_mm,
        'Bill Depth (mm)': bill_depth_mm,
        'Flipper Length (mm)': flipper_length_mm,
        'Body Mass (g)': body_mass_g,
        'Gender': gender,
        'Island': island
    }
    input_df = pd.DataFrame(input_data, index=[0])

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

# SVM Classification Metrics
with st.expander('Cross Validation'):
    st.caption('Train set 80%, Test set 20%; Sampling without replacement')
    # Prepare data for classification
    X_numeric = combined_df.drop('species', axis=1)
    y_numeric = combined_df['species']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_numeric, test_size=0.2, random_state=42)

    # Create and fit the SVM classifier
    svm_classifier = SVC(kernel='poly', probability=True)  # You can also try 'rbf' or other kernels
    svm_classifier.fit(X_train, y_train)

    # Make predictions
    y_train_pred = svm_classifier.predict(X_train)
    y_test_pred = svm_classifier.predict(X_test)

    # Calculate metrics
    from sklearn.metrics import accuracy_score
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy'],
        'Train': [train_accuracy],
        'Test': [test_accuracy]
    })

    # Set the Metric column as the index
    metrics_df.set_index('Metric', inplace=True)

    # Display the metrics in a table
    st.write('**Classification Metrics**')
    st.dataframe(metrics_df, use_container_width=False)

# Input features expander
with st.expander('Input Features'):
    st.write(input_df)  # Display the chosen input features in a beautiful table

# Model training and inference
X = X_numeric  # Ensure X is defined
y = y_numeric  # Ensure y is defined

# Train the classifier
clf = SVC(kernel='poly', probability=True) 
clf.fit(X, y)

# Prepare input for prediction
input_row = input_df.drop(columns=['Gender', 'Island'])  # Drop non-numeric columns
input_row = input_row.reindex(columns=X.columns, fill_value=0)  # Ensure columns match

# Make predictions
prediction_proba = clf.predict_proba(input_row)

# Get the index of the species with the highest probability
predicted_index = np.argmax(prediction_proba)

# Use that index to get the predicted species
predicted_species = list({'Adelie': 1, 'Chinstrap': 2, 'Gentoo': 3}.keys())[predicted_index]

# Display the predicted species
st.success(f"Predicted Species: {predicted_species}")

# Display prediction probabilities
df_prediction_proba = pd.DataFrame(prediction_proba, columns={'Adelie': 1, 'Chinstrap': 2, 'Gentoo': 3}.keys())
df_prediction_proba_percentage = df_prediction_proba * 100
df_prediction_proba_percentage = df_prediction_proba_percentage.round(2)


# Display probabilities in a nice DataFrame
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
