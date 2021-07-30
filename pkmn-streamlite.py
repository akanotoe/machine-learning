# Based on a tutorial at https://www.realpythonproject.com/build-a-streamlit-app-to-predict-if-a-pokemon-is-legendary/

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def title(s):
    st.text("")
    st.title(s)
    st.text("")

def clean_and_split(df):
    legendary_df = df[df['Legendary'] == True]
    normal_df = df[df['Legendary'] == False].sample(51)
    legendary_df.fillna(legendary_df.mean(), inplace = True)
    normal_df.fillna(normal_df.mean(), inplace = False)
    feature_list = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary', 'height (m)', 'weight (kg)']
    sub_df = pd.concat([legendary_df, normal_df])[feature_list]
    X = sub_df.loc[:, sub_df.columns != 'Legendary']
    Y = sub_df['Legendary']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1, test_size = 0.2, shuffle = True, stratify = Y)
    
    return X_train, X_test, y_train, y_test

# Intro
st.title('Is that a Legendary Pokémon?')
st.image('746Wishiwashi-Solo.png', width = 512)
# st.markdown('''
#     Photo by [Kamil S](https://unsplash.com/@16bitspixelz?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)
#     on [Unsplash](https://unsplash.com/s/photos/pokemon?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText).
# ''')

# Load data
df = pd.read_csv('pokemon_data.csv')
st.dataframe(df.head())

# Basic Info
shape = df.shape
num_total = len(df)
num_legendary = len(df[df['Legendary'] == True])
num_non_legendary = num_total - num_legendary

st.subheader('''
    Number of Pokémon: {}
'''.format(num_total))

st.subheader('''
    Number of Legendary Pokémon: {}
'''.format(num_legendary))

st.subheader('''
    Number of non-Legendary Pokémon: {}
'''.format(num_non_legendary))

st.subheader('''
    Number of features: {}
'''.format(shape[1]))

# Plot pokemon based on Type 1 field
title('Legendary Pokémon Distribution based on Type')
legendary_df = df[df['Legendary'] == True]
fig1 = plt.figure()
ax = sns.countplot(data = legendary_df, x = 'Type 1', order = legendary_df['Type 1'].value_counts().index)
plt.xticks(rotation = 45)
st.pyplot(fig1)

# Nice

title('Height vs. weight for Legendary and non-Legendary Pokémon')
fig2 = plt.figure()
sns.scatterplot(data = df, x = 'weight (kg)', y = 'height (m)', hue = 'Legendary')
st.pyplot(fig2)

# Heatmap correlation of features
title('Correlation between features')
st.subheader('Legendary Pokémon only')
st.text('')
fig3 = plt.figure()
sns.heatmap(legendary_df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'height (m)', 'weight (kg)']].corr())
st.pyplot(fig3)

# Attack vs. Sp. Atk
title('Attack vs. Special Attack')
fig4 = plt.figure()
sns.scatterplot(data = df, x = 'Sp. Atk', y = 'Attack', hue = 'Legendary')
st.pyplot(fig4)

# Random forest predictor
title('Random Forest')
X_train, X_test, y_train, y_test = clean_and_split(df)
st.subheader('Sample Data')
st.dataframe(X_train.head(3))
model = RandomForestClassifier(n_estimators = 100)
model.fit(X_train, y_train)

# Show scores of predictive model
title('Metrics')
st.subheader('Model Score: {}'.format(model.score(X_test, y_test)))
st.subheader('Precision Score: {}'.format(precision_score(model.predict(X_test), y_test)))
st.subheader('Recall Score: {}'.format(recall_score(model.predict(X_test), y_test)))

st.subheader('Confusion Matrix')
fig5 = plt.figure()
conf_matrix = confusion_matrix(model.predict(X_test), y_test)
sns.heatmap(conf_matrix, annot = True, xticklabels = ['Normal', 'Legendary'],
    yticklabels = ['Normal', 'Legendary'])
plt.ylabel('True')
plt.xlabel('Predicted')
st.pyplot(fig5)