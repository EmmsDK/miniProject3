import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv("/Users/emiliocastrolagunas/Desktop/Sem4/BI/Notebooks/Data/house-data.csv", sep=',')

st.title("Mini Project 3 Machine Learning for Prediction by Regression")

# Display basic information about the data
st.write(df.info())

st.title('Scatter Plot Example')

# Create the scatter plot
fig, ax = plt.subplots()
ax.scatter(df.sqft_living15, df.price, color='green')
ax.set_xlabel('sqft_living15')
ax.set_ylabel('price')

# Display the plot using st.pyplot()
st.pyplot(fig)

st.title('Distribution Plot Example')

# Create the distribution plot using Seaborn
fig2, ax2 = plt.subplots()
sns.distplot(df['sqft_living15'], label='sqft_living15', norm_hist=True)

# Set plot labels and title
ax2.set_xlabel('sqft_living15')
ax2.set_ylabel('Density')
ax2.set_title('Distribution of sqft_living15')

# Display the plot using st.pyplot()
st.pyplot(fig2)

# Create a correlation matrix from your data
selected_features = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']]

# Print the data types of the selected features
st.write(selected_features.dtypes)

numeric_features = selected_features.select_dtypes(include=['number'])

correlation_matrix = numeric_features.corr()

# Streamlit app
st.title('Heatmap Example')

# Create the heatmap using Seaborn
fig3, ax3 = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)

# Display the heatmap using st.pyplot() with the figure and axes objects
st.pyplot(fig3)
