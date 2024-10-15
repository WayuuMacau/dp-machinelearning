import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import altair as alt

st.title('🐧Penguin Classifier App')
st.subheader('🤖 Machine Learning Model - Support Vector Machine')
st.info('Designed by Lawrence Ma 🇭🇰 +852 55767752 or 🇲🇴 +853 62824370')
st.warning("Try to fine-tune the left-hand side parameters to see the prediction result of penguin species")

# Load and prepare data
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
X_raw = df.drop('species', axis=1)
y_raw = df.species

# Convert the target variable y to numeric
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
input_data = {
    'bill_length_mm': [bill_length_mm],
    'bill_depth_mm': [bill_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [gender],
    'island': [island]
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
    colors = {
        'Adelie': 'rgb(0, 0, 255)',       # Blue
        'Chinstrap': 'rgb(255, 165, 0)',  # Orange
        'Gentoo': 'rgb(0, 128, 0)'        # Green
    }
    df['color'] = df['species'].map(colors)

    st.caption('Below coordinates of red circles are the parameters chosen by left sidebar.')
    
    # 第一個散點圖
    scatter1 = alt.Chart(df).mark_circle(size=60).encode(
        x='bill_length_mm',
        y='body_mass_g',
        color=alt.Color('color:N', scale=None),
        tooltip=['species']
    ).interactive()

    red_circle1 = alt.Chart(input_df).mark_circle(size=100, color='red').encode(
        x='bill_length_mm',
        y='body_mass_g'
    )

    st.altair_chart(scatter1 + red_circle1, use_container_width=True)

    # 第二個散點圖
    scatter2 = alt.Chart(df).mark_circle(size=60).encode(
        x='bill_depth_mm',
        y='flipper_length_mm',
        color=alt.Color('color:N', scale=None),
        tooltip=['species']
    ).interactive()

    red_circle2 = alt.Chart(input_df).mark_circle(size=100, color='red').encode(
        x='bill_depth_mm',
        y='flipper_length_mm'
    )

    st.altair_chart(scatter2 + red_circle2, use_container_width=True)

    # 第三個散點圖
    scatter3 = alt.Chart(df).mark_circle(size=60).encode(
        x='bill_depth_mm',
        y='bill_length_mm',
        color=alt.Color('color:N', scale=None),
        tooltip=['species']
    ).interactive()

    red_circle3 = alt.Chart(input_df).mark_circle(size=100, color='red').encode(
        x='bill_depth_mm',
        y='bill_length_mm'
    )

    st.altair_chart(scatter3 + red_circle3, use_container_width=True)

    # 第四個散點圖
    scatter4 = alt.Chart(df).mark_circle(size=60).encode(
        x='flipper_length_mm',
        y='body_mass_g',
        color=alt.Color('color:N', scale=None),
        tooltip=['species']
    ).interactive()

    red_circle4 = alt.Chart(input_df).mark_circle(size=100, color='red').encode(
        x='flipper_length_mm',
        y='body_mass_g'
    )

    st.altair_chart(scatter4 + red_circle4, use_container_width=True)

# Correlation expander
with st.expander('Correlation'):
    # Ensure all data is numeric
    combined_df_numeric = combined_df.select_dtypes(include=[np.number])

    # Calculate correlation of each feature with the target variable
    correlation_with_y = combined_df_numeric.corr()['species'].drop('species')

    # Create a DataFrame for better display
    correlation_df = correlation_with_y.reset_index()
    correlation_df.columns = ['Feature', 'Correlation with y']

    # Set the Feature column as the index
    correlation_df.set_index('Feature', inplace=True)

    # Display the correlation table with narrow columns
    st.write('**Correlation between each feature and the target variable**')
    st.dataframe(correlation_df, use_container_width=False)

        
# SVC Classifier Metrics
with st.expander('SVC Classifier Metrics'):
    st.caption('Train set 80%, Test set 20%; Sampling without replacement')
    
    # Prepare data for classification
    X_numeric = combined_df_numeric.drop('species', axis=1)
    y_numeric = combined_df_numeric['species']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_numeric, test_size=0.2, random_state=42)

    # Create and fit the SVC classifier
    svc_classifier = SVC(kernel='rbf', probability=True)
    svc_classifier.fit(X_train, y_train)

    # Make predictions
    y_train_pred = svc_classifier.predict(X_train)
    y_test_pred = svc_classifier.predict(X_test)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
        'Train': [train_accuracy, train_precision, train_recall, train_f1],
        'Test': [test_accuracy, test_precision, test_recall, test_f1]
    })

    # Set the Metric column as the index
    metrics_df.set_index('Metric', inplace=True)

    # Display the metrics in a table
    st.write('**SVC Classifier Metrics**')
    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=False)

    # Add summary text
    st.markdown("""
    **Summary:**
    
    The SVC classifier shows good performance on both the training and test sets. 
    
    - Accuracy: The model correctly classifies a high percentage of samples in both sets, with slightly better performance on the training data.
    - Precision: The model has a high precision, indicating a low false positive rate.
    - Recall: The high recall suggests that the model is effective at identifying positive cases.
    - F1-score: The F1-score, being the harmonic mean of precision and recall, indicates a good balance between precision and recall.

    The close values between train and test metrics suggest that the model is generalizing well and not overfitting significantly. However, the slightly higher values for the training set indicate there might be a small amount of overfitting, which is common and often acceptable if not too large.
    """)
    
with st.expander('Input features'):
    st.caption('Below values are the parameters chosen by left sidebar.')
    # Set the Metric column as the index
    input_df.set_index('bill_length_mm', inplace=True)
    st.dataframe(input_df, use_container_width=False)  # Set to False for narrow width

st.header("", divider="rainbow")

# Data preparation
# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(pd.concat([input_df, X_raw], axis=0), prefix=encode)
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

# Get the index of the species with the highest probability
predicted_index = np.argmax(prediction_proba)

# Use that index to get the predicted species
predicted_species = list(target_mapper.keys())[predicted_index]

# Display the predicted species
st.success(f"Predicted Species: {predicted_species}")

# Display prediction probabilities
df_prediction_proba = pd.DataFrame(prediction_proba, columns=target_mapper.keys())
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
