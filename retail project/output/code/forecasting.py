import pandas as pd

df = pd.read_csv("Walmart_Sales.csv")
print(df.head())
print(df.info())


df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
df = df.dropna()

df['Month'] = df['Date'].dt.month
df['Dayofweek'] = df['Date'].dt.dayofweek



#select Features

X = df[['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Month']]

y = df['Weekly_Sales']


#Train AI model

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X,y)

df['Predicted_Sales'] = model.predict(X)

#inventory risk logic
df['Stock'] = df['Weekly_Sales'] * 1.1

df['Risk'] = df.apply(
    lambda x: "Understock" if x['Predicted_Sales'] > x['Stock'] else "Overstock",
    axis=1

    )

df.to_csv("final_walmart_data.csv", index=False)
