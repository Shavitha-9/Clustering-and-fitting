import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# Load the data into a pandas dataframe
data = pd.read_csv('Inflation_consumer_prices_(annual_%).csv')

# Select the columns we want to cluster
X = data.iloc[:, 1:]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster the data using K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Add the cluster labels to the dataframe
data['cluster'] = kmeans.labels_

# Plot the clusters
plt.scatter(data['United States'], data['United Kingdom'], c=data['cluster'])
plt.xlabel('United States')
plt.ylabel('United Kingdom')

# Plot the cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=150, linewidths=3, color='r')

# Save the plot as a PNG file
plt.savefig('Clustering.png')

# Extract data for each country
countries = ['United States', 'United Kingdom', 'Ukraine', 'Russian Federation', 'Japan', 'Korea, Rep.', 'India', 'China', 'Brazil', 'France', 'Germany']

# Create array of years
years = np.arange(2008, 2021)

# Define dark colors
colors = ['#FF8C00', '#800080', '#008B8B', '#8B0000', '#2E8B57', '#FF4500', '#1E90FF', '#FF1493', '#228B22', '#A0522D', '#4B0082']

# Perform polynomial regression for each country
fig, ax = plt.subplots(figsize=(10, 6))

# Initialize the y-axis limits
y_min = float('inf')
y_max = float('-inf')

for i, country in enumerate(countries):
    country_data = data[country]

    # Update y-axis limits based on the minimum and maximum values of the data
    y_min = min(y_min, np.min(country_data))
    y_max = max(y_max, np.max(country_data))

    # Fit polynomial regression to country's data
    coefficients = np.polyfit(years, country_data, 3)
    poly = np.poly1d(coefficients)

    # Predict CO2 emissions for next 10 years
    future_years = np.arange(2021, 2031)
    predicted_data = poly(future_years)

    # Plot data and fitted function with dark colors
    ax.plot(years, country_data, '-', label=country, color=colors[i % len(colors)])
    ax.plot(future_years, predicted_data, linestyle='dotted', color=colors[i % len(colors)])

# Set y-axis limits based on the minimum and maximum values of the data
ax.set_ylim(y_min, y_max)

ax.legend()
ax.set_xlabel('Year')
ax.set_ylabel('Inflation, consumer prices (annual %)')
# Set bold and colorful title
title_text = 'Inflation actual and predicted for year 2008 to 2030'
ax.set_title(title_text, fontweight='bold', color='Blue', fontsize=16)




# Save the plot as a PNG file
plt.savefig('Prediction.png')

# Show the plots
plt.show()
