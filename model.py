import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# load the dataset
df = pd.read_csv("mobile_price_range_data.csv")
df.head()

df.info()

plt.figure(figsize=(10,6))
# Plotting the distribution of battery power
sns.boxplot(x = "price_range",y="battery_power",data = df , color= "blue")
plt.title("Battery Power vs Price Range")
plt.show()

# I want to create a correlation matrix and visualixe it with a  heatmap
plt.figure(figsize=(12,8))
# Create a correlation matrix
correlation_matrix = df.corr()
# Create a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# now i want to see relationship between ram and price range
plt.figure(figsize=(10,6))
# Plotting the distribution of ram
sns.boxplot(x = "price_range",y="ram",data = df , color= "blue")
plt.title("RAM vs Price Range")
plt.show()

df.isnull().sum()
df.duplicated().sum()

# feature scaling >>> standardization
x = df.drop("price_range",axis=1)
y = df["price_range"]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

# now we do spliting of data into tran and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# now we create logistic , decision tree and svm model
#1. logistic model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
lt_model = LogisticRegression()
lt_model.fit(x_train,y_train)

y_pred = lt_model.predict(x_test)
print(classification_report(y_test,y_pred))
accuracy_score = lt_model.score(x_test,y_test)
print("Accuracy of Logistic Regression Model: ", accuracy_score)
# confusion matrix
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt="d",cmap="Blues")
plt.title("Confusion Matrix for Logistic Regression")  
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
