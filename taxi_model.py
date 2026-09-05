import pandas as pd
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io 
import base64

df = pd.read_csv("data/taxi_data.csv")
x = df[["distance_km"]]
y = df[["cost"]]

def calculate_cost(distance):
    result = model.predict([[distance]])[0][0]
    return round (result,2)

def get_num_records():
    return len(df)

def generate_plot():
    fig, ax = plt.subplots(figsize =(7,5))
    ax.scatter(df["distance_km"], df["cost"], alpha = 0.5, label = "Data" )
    ax.plot(df["distance_km"], model.predict(x), color = "red", linewith =2, label = "Regression line")
    ax.set_tittle("Taxi Fare vs Distance")
    ax.set_xlable("Distance (km)")
    ax.set_ylabel("Cost ($)")
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format = "png", bbox_inches = "tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
