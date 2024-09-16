import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import altair as alt

st.title('🐧Penguin Classifier App')
st.subheader('🤖 machine learning model - support vector machine')
st.info('Designed by Lawrence Ma 🇲🇴 +853 62824370 or 🇭🇰 +852 55767752')

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

with st.expander('Data visualization'):
    colors = {
        'Adelie': 'rgb(0, 0, 255)',       # Blue
        'Chinstrap': 'rgb(255, 165, 0)',  # Orange
        'Gentoo': 'rgb(0, 128, 0)'        # Green
    }
    df['color'] = df['species'].map(colors)
    scatter = alt.Chart(df).mark_circle(size=60).encode(
        x='bill_length_mm',
        y='body_mass_g',
        color=alt.Color('color:N', scale=None),
        tooltip=['species']
    ).interactive()
    st.altair_chart(scatter, use_container_width=True)

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
    st.write(input_df)

st.header("",divider="rainbow")

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
               ), 
             }, hide_index=True)
