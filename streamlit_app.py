import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC

st.title('🐧Penguin Classifier App')
st.subheader('🤖machine learning model - support vector machine')
st.info('Powered by Lawrence Ma 🇲🇴 +853 62824370 or 🇭🇰 +852 55767752')


with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.species
  y_raw

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Input features
with st.sidebar:
  st.header('Input features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 42.8)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 19.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 199.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4201.0)
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
  input_df


# Model training and inference
## Train the ML model
clf = SVC(kernel='poly', probability=True) 
clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={0: 'Adelie',
                                 1: 'Chinstrap',
                                 2: 'Gentoo'})

st.header("",divider="rainbow")

# Display predicted species
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
predicted_species = penguins_species[prediction][0]
st.success(f"**Predicted Species:** {predicted_species}")

#st.subheader('Predicted Species')
df_prediction_proba_percentage = df_prediction_proba * 100  # 將概率轉換為百分比
df_prediction_proba_percentage = df_prediction_proba_percentage.round(2)  # 四捨五入到小數點後兩位

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

#penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
#predicted_species = penguins_species[prediction][0]
#st.success(f"Predicted Species: {predicted_species}")
