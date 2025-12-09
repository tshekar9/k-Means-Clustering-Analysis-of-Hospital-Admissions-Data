# Data Cleaning


#import required packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# load data set into dataframe
df = pd.read_csv("./medical_clean.csv", index_col=0)

# overview of data
df.head()

#check shape of dataframe
df.shape

# check data types and verify there are no null values
df.info()

# drop irrelevant columns
df = df[["Age", "TotalCharge", "Additional_charges"]]

# rename column headings
df.rename(columns={
    "Age": "age",
    "TotalCharge" : "daily_charge",
    "Additional_charges": "additional_charges"
}, inplace=True)


#summary stats of selected variables - age, daily charge and additional charges
df.describe()

# distributions of selected variables - age, daily charge and additional charges
plt.figure(figsize=[20,5])
plt.subplot(1,3,1)
plt.hist(df["age"])
plt.title("Age Distribution (Uniform Distribution)")
plt.xlabel("Age")
plt.ylabel("Number of Patients")

plt.subplot(1,3,2)
plt.hist(df["daily_charge"])
plt.title("Distribution of Daily Charge (Bimodal)")
plt.xlabel("Daily Charge")
plt.ylabel("Number of Patients")

plt.subplot(1,3,3)
plt.hist(df["additional_charges"])
plt.title("Distribution of Additional Charges (Right-Skewed)")
plt.xlabel("Additional Charges")
plt.ylabel("Number of Patients")

# scale the data using Standard Scaler
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns = ["age", "daily_charge", "additional_charges"])

# overview of final scaled data set 
# Suppress scientific notation
pd.options.display.float_format = '{:.2f}'.format
scaled_df.head()

# scaled data has mean 0 and standard deviation 1
scaled_df.describe()


# Cleaned Data Set

# export cleaned and scaled data set as csv file
scaled_df.to_csv("scaled_medical.csv", index = False)




