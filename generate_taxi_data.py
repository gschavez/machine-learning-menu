import pandas as pd
import random

data=[]



for _ in range (600):

    distance = round(random.uniform(1,30),2)  #this part generates a random distance 
    cost = 5000 + (2500 * distance) + random.uniform(-1500,1500)
    data.append({
        "distance_km": distance,
        "cost": round(cost , 0)
    })

df = pd.DataFrame(data)

df.to_csv("data/taxi_data.csv", index=False)

print("Dataset generated succesfully.")
print(f"Number of record: {len(df)}")
print(df.head())

          


