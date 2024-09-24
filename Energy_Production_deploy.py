# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 00:37:23 2024

@author: Administrator
"""
"""
Streamlit App for Regression Model with Data Visualizations
"""
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# Load the saved regression model and scaler
with open("RandomForestRegressor.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Load the dataset
data = pd.read_csv("Copy of energy_production (1).csv", sep=';')

data = data.drop_duplicates()
print(data)

# Sidebar for user input
st.sidebar.header('User Input Features')

def user_input_features():
    temperature = st.sidebar.slider('Temperature (°C)', min_value=0.0, max_value=100.0, step=0.1)
    exhaust_vacuum = st.sidebar.slider('Exhaust Vacuum (cm Hg)', min_value=30.0, max_value=60.0, step=0.1)
    amb_pressure = st.sidebar.slider('Ambient Pressure (mbar)', min_value=1000.0, max_value=1050.0, step=0.1)
    r_humidity = st.sidebar.slider('Relative Humidity (%)', min_value=0.0, max_value=100.0, step=0.1)
    
    data_input = {
        'temperature': temperature,
        'exhaust_vacuum': exhaust_vacuum,
        'amb_pressure': amb_pressure,
        'r_humidity': r_humidity
    }
    features = pd.DataFrame(data_input, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(input_df)

# Predict the energy production
prediction = rf_model.predict(input_df)

# Display the prediction
st.subheader('Predicted Energy Production')
st.write(f"Predicted Energy Production: {prediction[0]:.2f}")


# Add data visualizations
st.subheader("Data Visualizations:")

# Create a selectbox to allow the user to select the feature to visualize
feature_select = st.selectbox("Select a feature to visualize:", ["temperature", "exhaust_vacuum", "amb_pressure", "r_humidity"])

# Create a figure with a scatter plot
fig = px.scatter(data, x=feature_select, y="energy_production", title=f"{feature_select.capitalize()} vs Energy Production")

# Update the graph based on the selected feature
st.plotly_chart(fig)

# Line plot for energy production over time
st.write("Energy Production Over Time:")
fig5 = px.line(data, y='energy_production', title='Energy Production Over Time (Index)')
st.plotly_chart(fig5)

# 2D Scatter plot of Temperature vs Energy Production, colored by Ambient Pressure
st.write("2D Scatter Plot (Temperature vs Energy Production, colored by Ambient Pressure):")
fig6 = px.scatter(data, x='temperature', y='energy_production', color='amb_pressure', title='Temperature vs Energy Production')
st.plotly_chart(fig6)

# Side by side histograms
st.write("Histograms:")
fig7 = make_subplots(rows=2, cols=2, subplot_titles=("Temperature", "Energy Production", "Exhaust Vacuum", "Relative Humidity"))

fig7.add_trace(go.Histogram(x=data['temperature'], nbinsx=50, name='Temperature'), row=1, col=1)
fig7.add_trace(go.Histogram(x=data['energy_production'], nbinsx=50, name='Energy Production'), row=1, col=2)
fig7.add_trace(go.Histogram(x=data['exhaust_vacuum'], nbinsx=50, name='Exhaust Vacuum'), row=2, col=1)
fig7.add_trace(go.Histogram(x=data['r_humidity'], nbinsx=50, name='Relative Humidity'), row=2, col=2)

fig7.update_layout(title_text="Side by Side Histograms", showlegend=False, height=600, width=800)
st.plotly_chart(fig7)

# Pair plot of all features
st.write("Scatter Matrix (Pair Plot) of Features:")
fig8 = px.scatter_matrix(data, dimensions=['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity', 'energy_production'], 
                         title='Scatter Matrix of Features', color='energy_production')
st.plotly_chart(fig8)

# Correlation matrix heatmap
st.write("Correlation Matrix:")
correlation_matrix = data.corr()
plt.figure(figsize=(14,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot(plt)

