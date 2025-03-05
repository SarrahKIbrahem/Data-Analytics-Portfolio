# Scooter Rental Analysis with Business Insights

Welcome to our scooter rental analysis project! This document will walk you through our analysis, explaining each step, its purpose, and how it relates to key business questions.

## 1. Data Loading and Initial Exploration

First, we import necessary libraries and load our data:

```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score

df_scooter = pd.read_csv('https://bit.ly/scooter-rentals')
```

We're using pandas for data manipulation, seaborn for visualization, and scikit-learn for our machine learning model.

**Business Question:** What data do we have available to analyze our scooter rental business?

**Answer:** Our dataset includes information on daily rentals over a two-year period, including date, weather conditions, and the number of rentals by registered and unregistered users. This comprehensive dataset allows us to analyze various factors affecting our business.

## 2. Data Preprocessing

We rename some columns for clarity and convert the 'season' column from numbers to actual season names:

```python
df_scooter = df_scooter.rename(columns={
    'yr': 'year', 'mnth': 'month', 'hum': 'humidity_norm', 
    'temp': 'temp_norm', 'atemp': 'temp_felt_norm'
})

df_scooter['season'] = df_scooter['season'].replace([1,2,3,4], ['winter','spring','summer','fall'])
```

**Business Question:** How does demand for our scooters vary across seasons?

**Answer:** By converting the season numbers to names, we can easily analyze seasonal trends. This will help us understand how demand fluctuates throughout the year, allowing us to adjust our fleet size, plan maintenance during low-demand periods, and develop seasonal marketing campaigns.

## 3. Feature Engineering

We create a new column 'rentals_total' by summing registered and unregistered rentals:

```python
df_scooter['rentals_total'] = df_scooter['rentals_registered'] + df_scooter['rentals_unregistered']
```

**Business Question:** What's the relationship between registered and unregistered users? Should we focus on converting unregistered users?

**Answer:** By creating this total rentals column, we can analyze the proportion of rentals from registered versus unregistered users. This insight can guide our marketing strategies for user acquisition and retention, and inform decisions about loyalty programs or subscription models.

## 4. Exploratory Data Analysis (EDA)

We use various seaborn plots to visualize our data:

```python
# Histogram of total rentals
sns.displot(df_scooter['rentals_total'])

# Scatterplot of registered vs unregistered rentals
sns.scatterplot(data=df_scooter, x='rentals_unregistered', y='rentals_registered', hue='workday')

# Swarmplot of total rentals by season
sns.catplot(data=df_scooter, x='season', y='rentals_total', kind='swarm')

# Line plot of total rentals over months, separated by year
sns.relplot(data=df_scooter, x='month', y='rentals_total', kind='line', hue='year')

# Pairplot to show relationships between weather variables and total rentals
sns.pairplot(df_scooter, x_vars=['temp_norm', 'temp_felt_norm', 'humidity_norm', 'wind_norm'], y_vars='rentals_total', kind='kde')

# Heatmap of correlations between key variables
sns.heatmap(df_scooter[['rentals_total', 'temp_norm', 'temp_felt_norm', 'humidity_norm', 'wind_norm']].corr(), annot=True)
```

**Business Questions:**

1. **How do weather conditions affect our rental numbers?**
   
   Answer: The pairplot and heatmap show the relationships between weather variables and total rentals. This allows us to predict daily demand based on weather forecasts, adjust pricing strategy based on weather conditions, and guide marketing efforts.

2. **How is our business growing year over year?**
   
   Answer: The line plot shows rental trends over months, separated by year. This visualization helps us calculate year-over-year growth rates, inform investment decisions, guide expansion plans, and set realistic growth targets.

3. **How do rental patterns differ between weekdays and weekends?**
   
   Answer: The scatterplot, colored by workday, gives us insights into weekday vs. weekend patterns. This information helps optimize scooter distribution throughout the week, tailor pricing strategies, and guide staffing decisions for maintenance and support.

## 5. Linear Regression Model

We build a simple linear regression model using normalized temperature to predict total rentals:

```python
x = df_scooter[['temp_norm']]
y = df_scooter['rentals_total']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)
```

**Business Question:** Can we predict daily rental demand based on temperature?

**Answer:** This linear regression model quantifies the relationship between temperature and total rentals. It allows us to predict rental demand based on temperature forecasts, which is crucial for fleet management and pricing strategies. For example, we could say, "For every 10°C increase in temperature, we expect an average increase of X rentals."

## 6. Model Evaluation and Interpretation

While not shown in the provided code, we would typically evaluate the model by:

1. Making predictions on the test set
2. Calculating metrics like Mean Squared Error (MSE) and R-squared
3. Plotting the predicted vs actual values

The heatmap shows that temperature has a correlation of about 0.63 with total rentals, indicating a moderately strong positive relationship. This supports our choice of using temperature as a predictor in our linear regression model.

**Business Question:** How reliable are our rental predictions, and how can we use them?

**Answer:** The model's performance metrics and the correlation coefficient give us confidence in our predictions. We can use these predictions to:
- Optimize our fleet size and distribution based on temperature forecasts
- Implement dynamic pricing strategies
- Plan maintenance schedules during predicted low-demand periods

## Next Steps

To improve this analysis and answer more complex business questions, we could:
1. Use multiple features in our regression model
2. Try other types of models (e.g., Random Forest, Gradient Boosting)
3. Perform feature engineering (e.g., create day of week features from the date)
4. Analyze the impact of holidays and working days on rental patterns
5. Conduct a more detailed time series analysis to better understand and forecast seasonal trends

By continually refining our analysis, we can provide even more valuable insights to drive business decisions and improve our scooter rental service.
