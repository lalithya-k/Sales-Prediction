# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp
import warnings
import datetime

# Suppressing warnings
warnings.filterwarnings("ignore")

# Importing WordCloud for visualization
from wordcloud import WordCloud

# Importing scikit-learn modules for preprocessing and modeling
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Reading the dataset
df = pd.read_csv("C:\\Users\\konet\\Downloads\\supermarket-sales.csv")

# Displaying the first few rows of the dataframe
print(df.head())

# Displaying the shape and information of the dataframe
print(df.shape)
print(df.info())

# Descriptive statistics of the dataframe
print(df.describe())

# Unique values and value counts of 'Customer type' column
print(df['Customer type'].nunique())
print(df['Customer type'].value_counts())

# Value counts of categorical columns
print(df['Branch'].value_counts())
print(df['City'].value_counts())
print(df['Product line'].value_counts())
print(df['Payment'].value_counts())

# Creating a WordCloud for 'Product line'
plt.subplots(figsize=(20,8))
wordcloud = WordCloud(background_color='White',width=1920,height=1080).generate(" ".join(df['Product line']))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('cast.png')
plt.show()

# Checking for missing values in the dataframe
print(df.isnull().sum())

# Visualizing relationships between numeric variables
sns.scatterplot(data=df, x='Unit price', y='Rating',hue='Gender',style='Customer type')
plt.show()

# Visualizing distributions and outliers
plt.figure(figsize=(14,10))
sns.set_style(style='whitegrid')
plt.subplot(2,3,1)
sns.boxplot(x='Unit price',data=df)
plt.subplot(2,3,2)
sns.boxplot(x='Quantity',data=df)
plt.subplot(2,3,3)
sns.boxplot(x='Total',data=df)
plt.subplot(2,3,4)
sns.boxplot(x='cogs',data=df)
plt.subplot(2,3,5)
sns.boxplot(x='Rating',data=df)
plt.subplot(2,3,6)
sns.boxplot(x='gross income',data=df)
plt.show()

# Visualizing distributions using Kernel Density Estimation (KDE)
plt.figure(figsize=(14,10))
sns.set_style(style='whitegrid')
plt.subplot(2,3,1)
sns.kdeplot(x='Unit price',data=df)
plt.subplot(2,3,2)
sns.kdeplot(x='Quantity',data=df)
plt.subplot(2,3,3)
sns.kdeplot(x='Total',data=df)
plt.subplot(2,3,4)
sns.kdeplot(x='cogs',data=df)
plt.subplot(2,3,5)
sns.kdeplot(x='Rating',data=df)
plt.subplot(2,3,6)
sns.kdeplot(x='gross income',data=df)
plt.show()

# Pairplot for pairwise relationships between variables
sns.pairplot(data=df)
plt.show()

# Barplots to visualize relationships between Rating and Unit Price, and Quantity
plt.style.use("default")
plt.figure(figsize=(5,5))
sns.barplot(x="Rating", y="Unit price", data=df[170:180])
plt.title("Rating vs Unit Price",fontsize=15)
plt.xlabel("Rating")
plt.ylabel("Unit Price")
plt.show()

plt.style.use("default")
plt.figure(figsize=(5,5))
sns.barplot(x="Rating", y="Quantity", data=df[170:180])
plt.title("Rating vs Quantity",fontsize=15)
plt.xlabel("Rating")
plt.ylabel("Quantity")
plt.show()

# Calculating and visualizing correlation matrix
numeric_df = df.select_dtypes(include=np.number)
correlation_matrix = numeric_df.corr(method='spearman')
print(correlation_matrix)

plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True)
plt.show()

# Encoding categorical variables
list_1 = list(df.columns)
list_cate=[]
for i in list_1:
    if df[i].dtype=='object':
        list_cate.append(i)

le = LabelEncoder()
for i in list_cate:
    df[i] = le.fit_transform(df[i])

print(df)

# Splitting data into train and test sets
y = df['Gender']
x = df.drop('Gender',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

# Training KNN classifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)

# Checking shapes of train and test sets
print("Shape of x_train:", x_train.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

print("KNN classifier trained successfully.")
print("Number of neighbors:", knn.n_neighbors)

# Making predictions using KNN classifier
y_pred1=knn.predict(x_test)
print("Classification Report is:\n",classification_report(y_test,y_pred1))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred1))
print("Training Score:\n",knn.score(x_train,y_train)*100)

# Decision Tree Classifier
dtree = DecisionTreeClassifier(max_depth=6, random_state=123,criterion='entropy')

# Train the Decision Tree model using the training sets
dtree.fit(x_train,y_train)

# Predict the response for test dataset
y_pred2=dtree.predict(x_test)
print("Classification Report is:\n",classification_report(y_test,y_pred2))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred2))
print("Training Score:\n",dtree.score(x_train,y_train)*100)

# Random Forest Classifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)

# Predict the response for test dataset
y_pred3=rfc.predict(x_test)
print("Classification Report is:\n",classification_report(y_test,y_pred3))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred3))
print("Training Score:\n",rfc.score(x_train,y_train)*100)

# Gradient Boosting Classifier
gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)

# Predict the response for test dataset
y_pred4=gbc.predict(x_test)
print("Classification Report is:\n",classification_report(y_test,y_pred4))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred4))
print("Training Score:\n",gbc.score(x_train,y_train)*100)
