import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics 

#load dataset
df=pd.read_csv('cancer_data.csv')
df=df.dropna(axis=1)
count=df.diagnosis.value_counts()
mapping={'M':1, 'B':0}
df['diagnosis']=df['diagnosis'].map(mapping)
x=df.drop(['id', 'diagnosis'], axis='columns')
y=df.diagnosis.values

#train _test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20, random_state=0)

#evaluate models
from sklearn import tree
m1=tree.DecisionTreeClassifier(random_state=0)
m1.fit(x_train, y_train)
ms1=m1.score(x_test, y_test)
mp1=m1.predict(x_test)
accuracy = metrics.accuracy_score(y_test, mp1)
precision = metrics.precision_score(y_test, mp1)
sensitivity = metrics.recall_score(y_test, mp1)
specificity = metrics.recall_score(y_test, mp1, pos_label=0)
auc_roc = metrics.roc_auc_score(y_test, mp1)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("AUC ROC:", auc_roc)
input_data=x.iloc[0, 0:33]

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = m1.predict(input_data_reshaped)
#RF model
from sklearn.ensemble import RandomForestClassifier
m2=RandomForestClassifier(random_state=0)
m2.fit(x_train, y_train)
ms2=m2.score(x_test, y_test)
mp2=m2.predict(x_test)
accuracy = metrics.accuracy_score(y_test, mp2)
precision = metrics.precision_score(y_test, mp2)
sensitivity = metrics.recall_score(y_test, mp2)
specificity = metrics.recall_score(y_test, mp2, pos_label=0)
auc_roc = metrics.roc_auc_score(y_test, mp2)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("AUC ROC:", auc_roc)
input_data=x.iloc[0, 0:33]

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = m2.predict(input_data_reshaped)
#SVM model
from sklearn.svm import SVC
m3=SVC(random_state=0)
m3.fit(x_train, y_train)
ms3=m3.score(x_test, y_test)
mp3=m3.predict(x_test)
accuracy = metrics.accuracy_score(y_test, mp3)
precision = metrics.precision_score(y_test, mp3)
sensitivity = metrics.recall_score(y_test, mp3)
specificity = metrics.recall_score(y_test, mp3, pos_label=0)
auc_roc = metrics.roc_auc_score(y_test, mp3)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("AUC ROC:", auc_roc)
input_data=x.iloc[0, 0:33]

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = m3.predict(input_data_reshaped)
#LR model
from sklearn.linear_model import LogisticRegression
m4=LogisticRegression(random_state=0)
m4.fit(x_train, y_train)
ms4=m4.score(x_test, y_test)
mp4=m4.predict(x_test)
accuracy = metrics.accuracy_score(y_test, mp4)
precision = metrics.precision_score(y_test, mp4)
sensitivity = metrics.recall_score(y_test, mp4)
specificity = metrics.recall_score(y_test, mp4, pos_label=0)
auc_roc = metrics.roc_auc_score(y_test, mp4)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("AUC ROC:", auc_roc)
input_data=x.iloc[0, 0:33]

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = m4.predict(input_data_reshaped)
#print(prediction)
plt.figure(figsize=(10,10))
a=['Decision tree', 'Random Forest', 'SVM Classifier', 'Logistic Regression']
b=[ms1, ms2, ms3, ms4]
plt.bar(a,b)
plt.title('Graph for models vs accuracy')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
#plt.show()
##Confusion matrix for model DT

from sklearn.metrics import confusion_matrix
conf_matrix=confusion_matrix(y_test, mp1)
sns.heatmap(conf_matrix, annot=True, fmt=".0f")
plt.xlabel('predicted')
plt.ylabel('Actual')
plt.title('Decision tree confusion matrix')
#plt.show()
##Confusion matrix for model RF
conf_matrix=confusion_matrix(y_test, mp2)
sns.heatmap(conf_matrix, annot=True, fmt=".0f")
plt.xlabel('predicted')
plt.ylabel('Actual')
plt.title('Decision tree confusion matrix')
##Confusion matrix for model SVM
conf_matrix=confusion_matrix(y_test, mp3)
sns.heatmap(conf_matrix, annot=True, fmt=".0f")
plt.xlabel('predicted')
plt.ylabel('Actual')
plt.title('Decision tree confusion matrix')
##Confusion matrix for model LR
conf_matrix=confusion_matrix(y_test, mp4)
sns.heatmap(conf_matrix, annot=True, fmt=".0f")
plt.xlabel('predicted')
plt.ylabel('Actual')
plt.title('Decision tree confusion matrix')

#Classification Report
from sklearn.metrics import classification_report
report_DT=classification_report(y_test, mp1)
report_RF=classification_report(y_test, mp2)
report_SVM=classification_report(y_test, mp3)
report_LR=classification_report(y_test, mp4)
#print(report_dt, report_RF,report_SVM, report_LR)

###Streamlit deploy
import pickle
import sklearn
import streamlit as st
#input_data=[13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]
def cancer_pred(input_data):
  input_data_as_numpy_array = np.asarray(input_data)

# Using random Forest method. Because RF method gives good predicton score than other methods
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

  prediction = m2.predict(input_data_reshaped)
  print(prediction)

  if (prediction[0] == 0):
    return 'The person have "Benign" and is not diabetic'
  elif (prediction[0]==1):
    return 'The person have "Malignant" and is diabetic'
def main():
  st.title('Breacst cancer prediction')
  radius_mean=st.text_input('radius_mean')
  texture_mean=st.text_input('texture_mean')
  perimeter_mean=st.text_input('perimeter_mean')
  area_mean=st.text_input('area_mean')
  smoothness_mean=st.text_input('smoothness_mean')
  compactness_mean=st.text_input('compactness_mean')
  concavity_mean=st.text_input('concavity_mean')
  concave_points_mean=st.text_input('concave points_mean')
  symmetry_mean=st.text_input('symmetry_mean')
  fractal_dimension_mean=st.text_input('fractal_dimension_mean')
  radius_se=st.text_input('radius_se')
  texture_se=st.text_input('texture_se')
  
  perimeter_se=st.text_input('perimeter_se')
  area_se=st.text_input('area_se')
  smoothness_se=st.text_input('smoothness_se')
  compactness_se=st.text_input('compactness_se')
  concavity_se=st.text_input('concavity_se')
  concave_points_se=st.text_input('concave points_se')
  symmetry_se=st.text_input('symmetry_se')
  fractal_dimension_se=st.text_input('fractal_dimension_se')
  radius_worst=st.text_input('radius_worst')
  texture_worst=st.text_input('texture_worst')
  
  perimeter_worst=st.text_input('perimeter_worst')
  area_worst=st.text_input('area_worst')
  smoothness_worst=st.text_input('smoothness_worst')
  compactness_worst=st.text_input('compactness_worst')
  concavity_worst=st.text_input('concavity_worst')
  concave_points_worst=st.text_input('concave points_worst')
  symmetry_worst=st.text_input('symmetry_worst')
  fractal_dimension_worst=st.text_input('fractal_dimension_worst')

  diagnosis=''
  if st.button('test results'):
    diagnosis=cancer_pred([radius_mean, texture_mean, perimeter_mean,
       area_mean,smoothness_mean, compactness_mean, concavity_mean,
       concave_points_mean, symmetry_mean, fractal_dimension_mean,
       radius_se, texture_se, perimeter_se, area_se, smoothness_se, 
       compactness_se, concavity_se, concave_points_se, symmetry_se, 
       fractal_dimension_se, radius_worst, texture_worst, 
       perimeter_worst, area_worst, smoothness_worst, 
       compactness_worst, concavity_worst, concave_points_worst,
       symmetry_worst, fractal_dimension_worst])
  st.success(diagnosis)

if __name__== '__main__':
 main()
