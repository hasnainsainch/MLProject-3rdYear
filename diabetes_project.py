#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[ ]:


# 1️ Load Dataset
df = pd.read_csv("diabetes.csv")


# In[3]:


print("First 5 Rows:")
print(df.head())


# In[ ]:


# 2️ Data Preprocessing
# Replace 0 values in important columns with median
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


# In[5]:


for col in cols:
    df[col] = df[col].replace(0, df[col].median())


# In[6]:


# Features & Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# In[7]:


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[8]:


# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


# 3 Model Training
model = LogisticRegression()
model.fit(X_train, y_train)


# In[ ]:


# 4️  Prediction
y_pred = model.predict(X_test)


# In[11]:


# 5️⃣ Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[12]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

