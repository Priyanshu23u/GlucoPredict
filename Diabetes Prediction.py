#!/usr/bin/env python
# coding: utf-8

# # Pima Indian Diabetes Prediction

# In[1]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#loading the dataset
df = pd.read_csv("diabetes.csv")
df.head()


# ## Data Preprocessing

# In[3]:


#shape of the dataset
df.shape


# Checking the unique values for each variable in the dataset

# In[4]:


#checking unique values
variables = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
for i in variables:
    print(df[i].unique())


# In[5]:


variables = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age',]
for i in variables:
    c = 0
    for x in (df[i]):
        if x == 0:
            c = c + 1
    print(i,c)


# In[6]:


#replacing the missing values with the mean
variables = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for i in variables:
    df[i].replace(0,df[i].mean(),inplace=True)


# In[7]:


#checking to make sure that incorrect values are replace
for i in variables:
    c = 0
    for x in (df[i]):
        if x == 0:
            c = c + 1
    print(i,c)


# In[8]:


#missing values
df.info()


# In[9]:


#checking descriptive statistics
df.describe()


# In[10]:


df.head()


# In[51]:


plt.figure(figsize=(5,5))
plt.pie(df['Outcome'].value_counts(), labels=['No Diabetes', 'Diabetes'], autopct='%1.1f%%', shadow=False, startangle=90)
plt.title('Diabetes Outcome')
plt.show()


# In[16]:


sns.catplot(x="Outcome", y="Age", kind="swarm", data=df)


# In[48]:


fig,ax = plt.subplots(1,2,figsize=(15,5))
sns.boxplot(x='Outcome',y='Pregnancies',data=df,ax=ax[0])
sns.violinplot(x='Outcome',y='Pregnancies',data=df,ax=ax[1])


# ### Glucose and Diabetes

# In[27]:


sns.boxplot(x='Outcome', y='Glucose', data=df).set_title('Glucose vs Diabetes')


# ### Blood Pressuse and Diabetes

# In[35]:


fig,ax = plt.subplots(1,2,figsize=(15,5))
sns.boxplot(x='Outcome', y='BloodPressure', data=df, ax=ax[0]).set_title('BloodPressure vs Diabetes')
sns.violinplot(x='Outcome', y='BloodPressure', data=df, ax=ax[1]).set_title('BloodPressure vs Diabetes')


# ### Skin Thickness and Diabetes

# In[37]:


fig,ax = plt.subplots(1,2,figsize=(15,5))
sns.boxplot(x='Outcome', y='SkinThickness', data=df,ax=ax[0]).set_title('SkinThickness vs Diabetes')
sns.violinplot(x='Outcome', y='SkinThickness', data=df,ax=ax[1]).set_title('SkinThickness vs Diabetes')


# ### Insulin and Diabetes

# In[40]:


fig,ax = plt.subplots(1,2,figsize=(15,5))
sns.boxplot(x='Outcome',y='Insulin',data=df,ax=ax[0]).set_title('Insulin vs Diabetes')
sns.violinplot(x='Outcome',y='Insulin',data=df,ax=ax[1]).set_title('Insulin vs Diabetes')


# ### BMI and Diabetes

# In[47]:


fig,ax = plt.subplots(1,2,figsize=(15,5))
sns.boxplot(x='Outcome',y='BMI',data=df,ax=ax[0])
sns.violinplot(x='Outcome',y='BMI',data=df,ax=ax[1])


# ### Diabetes Pedigree Function and Diabetes Outcome

# In[43]:


fig,ax = plt.subplots(1,2,figsize=(15,5))
sns.boxplot(x='Outcome',y='DiabetesPedigreeFunction',data=df,ax=ax[0]).set_title('Diabetes Pedigree Function')
sns.violinplot(x='Outcome',y='DiabetesPedigreeFunction',data=df,ax=ax[1]).set_title('Diabetes Pedigree Function')


# ### Coorelation Matrix Heatmap

# In[52]:


#correlation heatmap
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm').set_title('Correlation Heatmap')


# ## Train Test Split

# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Outcome',axis=1),df['Outcome'],test_size=0.2,random_state=42)


# ## Diabetes Prediction

# ### Logistic Regression

# In[54]:


#building model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr


# In[56]:


#training the model
lr.fit(X_train,y_train)
#training accuracy
lr.score(X_train,y_train)


# In[57]:


#predicted outcomes
lr_pred = lr.predict(X_test)


# ### Random Forest Classifier

# In[76]:


#buidling model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,random_state=42)
rfc


# In[77]:


#training model
rfc.fit(X_train, y_train)
#training accuracy
rfc.score(X_train, y_train)


# In[78]:


#predicted outcomes
rfc_pred = rfc.predict(X_test)


# ### Support Vector Machine (SVM)

# In[79]:


#building model
from sklearn.svm import SVC
svm = SVC(kernel='linear', random_state=0)
svm


# In[80]:


#training the model
svm.fit(X_train, y_train)
#training the model
svm.score(X_test, y_test)


# In[81]:


#predicting outcomes
svm_pred = svm.predict(X_test)


# ## Model Evaluation

# ### Evaluating Logistic Regression Model

# #### Confusion Matrix Heatmap

# In[82]:


from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, cmap='Blues')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()


# #### Distribution plot

# In[83]:


ax = sns.distplot(y_test, color='r',  label='Actual Value',hist=False)
sns.distplot(lr_pred, color='b', label='Predicted Value',hist=False,ax=ax)
plt.title('Actual vs Predicted Value Logistic Regression')
plt.xlabel('Outcome')
plt.ylabel('Count')


# #### Classification Report

# In[84]:


from sklearn.metrics import classification_report
print(classification_report(y_test, lr_pred))


# The model has as an average f1 score of 0.755 and acuuracy of 78%.

# In[93]:


from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error,r2_score
print('Accuracy Score: ',accuracy_score(y_test,lr_pred))
print('Mean Absolute Error: ',mean_absolute_error(y_test,lr_pred))
print('Mean Squared Error: ',mean_squared_error(y_test,lr_pred))
print('R2 Score: ',r2_score(y_test,lr_pred))


# ### Evaluating Random Forest Classifier

# #### Confusion Matrix Heatmap

# In[85]:


sns.heatmap(confusion_matrix(y_test, rfc_pred), annot=True, cmap='Blues')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()


# #### Distribution Plot

# In[86]:


ax = sns.distplot(y_test, color='r',  label='Actual Value',hist=False)
sns.distplot(rfc_pred, color='b', label='Predicted Value',hist=False,ax=ax)
plt.title('Actual vs Predicted Value Logistic Regression')
plt.xlabel('Outcome')
plt.ylabel('Count')


# #### Classification Report

# In[87]:


print(classification_report(y_test, rfc_pred))


# The model has as an average f1 score of 0.745 and acuuracy of 77% which less in comparison to Logistic Regression model.

# In[94]:


print('Accuracy Score: ',accuracy_score(y_test,rfc_pred))
print('Mean Absolute Error: ',mean_absolute_error(y_test,rfc_pred))
print('Mean Squared Error: ',mean_squared_error(y_test,rfc_pred))
print('R2 Score: ',r2_score(y_test,rfc_pred))


# ### Evaluating SVM Model

# #### Confusion Matrix Heatmap

# In[88]:


sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, cmap='Blues')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()


# #### Distribution Plot

# In[89]:


ax = sns.distplot(y_test, color='r',  label='Actual Value',hist=False)
sns.distplot(svm_pred, color='b', label='Predicted Value',hist=False,ax=ax)
plt.title('Actual vs Predicted Value Logistic Regression')
plt.xlabel('Outcome')
plt.ylabel('Count')


# #### Classification Report

# In[90]:


print(classification_report(y_test, rfc_pred))


# The model has as an average f1 score of 0.745 and acuuracy of 77% which is equivalent to previous model.

# In[95]:


print('Accuracy Score: ',accuracy_score(y_test,svm_pred))
print('Mean Absolute Error: ',mean_absolute_error(y_test,svm_pred))
print('Mean Squared Error: ',mean_squared_error(y_test,svm_pred))
print('R2 Score: ',r2_score(y_test,svm_pred))


# ### Comparing the models

# In[97]:


#comparing the accuracy of different models
sns.barplot(x=['Logistic Regression', 'RandomForestClassifier', 'SVM'], y=[0.7792207792207793,0.7662337662337663,0.7597402597402597])
plt.xlabel('Classifier Models')
plt.ylabel('Accuracy')
plt.title('Comparison of different models')


# ## Conclusion

# From the exploratory data analysis, I have concluded that the risk of diabetes depends upon the following factors:
# 1. Glucose level
# 2. Number of pregnancies
# 3. Skin Thickness
# 4. Insulin level
# 5. BMI
# 
# With in increase in Glucose level, insulin level, BMI and number of pregnancies, the risk of diabetes increases. However, the number of pregnancies have strange effect of risk of diabetes which couldn't be explained by the data. The risk of diabetes also increases with increase in skin thickness.
# 
# Coming to the classification models, Logistic Regression outperformed Random Forest and SVM with 78% accuracy. The accuracy of the model can be improved by increasing the size of the dataset. The dataset used for this project was very small and had only 768 rows.
