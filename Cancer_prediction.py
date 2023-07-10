import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn import metrics
import streamlit as st

df=pd.read_csv('cancer_data.csv')
#check null value
df.isnull().sum()
df=df.dropna(axis=1)
print(df)
#check datatypes
print(df.dtypes)
mapping={'M':1, 'B':0}
df['diagnosis']=df['diagnosis'].map(mapping)

X=df.drop(['id', 'diagnosis'], axis='columns')
y=df.diagnosis.values

#train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#feature selection
feature_selection_methods = [
    ('Chi-Squared', SelectKBest(score_func=chi2, k=10)),
    ('ANOVA F-Value', SelectKBest(score_func=f_classif, k=10)),
    ('Mutual Information', SelectKBest(score_func=mutual_info_classif, k=10)),
    ("Recursive Feature Elimination (RFE)", RFE(estimator=LogisticRegression(), n_features_to_select=10)),
    ("L1-Regularization (Lasso)", SelectFromModel(estimator=LogisticRegression(penalty='l1', solver='liblinear'), threshold='median'))
    
]


classifiers = [
    LogisticRegression(),
    RandomForestClassifier(),
    SVC()
]


# Initialize lists to store accuracy values
method_names = []
classifier_names = []
accuracies = []

# Iterate over feature selection methods and classifiers
for method_name, method in feature_selection_methods:
    X_train_selected = method.fit_transform(X_train, y_train)
    X_test_selected = method.transform(X_test)

    for classifier in classifiers:
        model = classifier
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        # Append accuracy, method name, and classifier name to lists
        accuracies.append(accuracy)
        method_names.append(method_name)
        classifier_names.append(classifier.__class__.__name__)

# Create a DataFrame from the lists
df = pd.DataFrame({'Method': method_names, 'Classifier': classifier_names, 'Accuracy': accuracies})

# Create a grouped bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Method', y='Accuracy', hue='Classifier', data=df)
plt.xlabel('Feature Selection Method')
plt.ylabel('Accuracy')
plt.title('Comparison of Feature Selection Methods and Classifiers')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


#selector_lasso = SelectFromModel(estimator=LogisticRegression(penalty='l1', solver='liblinear'))
#selector_chi2 = SelectKBest(score_func=chi2, k=10)
selector_ANOVA= SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector_ANOVA.fit_transform(X_train, y_train)
X_test_selected=selector_ANOVA.transform(X_test)
# Get the indices of the selected features
selected_feature_indices = selector_ANOVA.get_support(indices=True)

# Get the names of the selected features
selected_features = [X_train.columns[idx] for idx in selected_feature_indices]

# Print the selected features
print("Selected Features:")
for feature in selected_features:
    print(feature)

selected_features=X[selected_features]
print(selected_features)

#model evaluation


# Train SVM model using the selected features
model1 = SVC()
model1.fit(X_train_selected, y_train)
model2 = LogisticRegression()
model2.fit(X_train_selected, y_train)
# Train Random Forest model using the selected features
model3 = RandomForestClassifier()
model3.fit(X_train_selected, y_train)



# Make predictions on the test data
svm_pred = model1.predict(X_test_selected)
lr_pred = model2.predict(X_test_selected)
rf_pred = model3.predict(X_test_selected)

# Evaluate SVM model performance
svm_accuracy = metrics.accuracy_score(y_test, svm_pred)
svm_precision = metrics.precision_score(y_test, svm_pred)
svm_recall = metrics.recall_score(y_test, svm_pred)
svm_f1_score = metrics.f1_score(y_test, svm_pred)
svm_auc = metrics.roc_auc_score(y_test, svm_pred)

lr_accuracy = metrics.accuracy_score(y_test, lr_pred)
lr_precision = metrics.precision_score(y_test, lr_pred)
lr_recall = metrics.recall_score(y_test, lr_pred)
lr_f1_score = metrics.f1_score(y_test, lr_pred)
lr_auc = metrics.roc_auc_score(y_test, lr_pred)
# Evaluate Random Forest model performance
rf_accuracy = metrics.accuracy_score(y_test, rf_pred)
rf_precision = metrics.precision_score(y_test, rf_pred)
rf_recall = metrics.recall_score(y_test, rf_pred)
rf_f1_score = metrics.f1_score(y_test, rf_pred)
rf_auc = metrics.roc_auc_score(y_test, rf_pred)
svm_classification_rep = classification_report(y_test, svm_pred)
lr_classification_rep = classification_report(y_test, lr_pred)
rf_classification_rep = classification_report(y_test, rf_pred)
# Print the evaluation metrics for SVM
print("SVM Accuracy:", svm_accuracy)
print("SVM Precision:", svm_precision)
print("SVM Recall:", svm_recall)
print("SVM F1-Score:", svm_f1_score)
print("SVM AUC-ROC:", svm_auc)
print('SVM Classification report', svm_classification_rep)
# Print the evaluation metrics for Random Forest
print("lr Accuracy:", lr_accuracy)
print("lr Precision:", lr_precision)
print("lr Recall:", lr_recall)
print("lr F1-Score:", lr_f1_score)
print("lr AUC-ROC:", lr_auc)
print('lr Classification report', lr_classification_rep)

# Print the evaluation metrics for Random Forest
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Precision:", rf_precision)
print("Random Forest Recall:", rf_recall)
print("Random Forest F1-Score:", rf_f1_score)
print("Random Forest AUC-ROC:", rf_auc)
print('RF Classification report',rf_classification_rep)


X_selected = selected_features

print(X_selected)
#selected_features = ['feature1', 'feature2', 'feature3']  # Replace with the desired feature names
input_data = X_selected.iloc[1, :]

# Convert input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make prediction
prediction = model2.predict(input_data_reshaped)

if (prediction[0] == 0):
    print('The person have "Benign" and is not diabetic')
elif (prediction[0]==1):
    print('The person have "Malignant" and is diabetic')



##compare features with original and selected
# Train and evaluate the model using all features
#model_all = LogisticRegression()
model2.fit(X_train, y_train)
y_pred_all = model2.predict(X_test)
accuracy_all = metrics.accuracy_score(y_test, y_pred_all)
print("Accuracy with all features:", accuracy_all)

# Train and evaluate the model using selected features
#model_selected = LogisticRegression()
model2.fit(X_train_selected, y_train)
#X_test_selected = selector_ANOVA.transform(X_test)
y_pred_selected = model2.predict(X_test_selected)
accuracy_selected = metrics.accuracy_score(y_test, y_pred_selected)
print("Accuracy with selected features:", accuracy_selected)

model2 = LogisticRegression()
model2.fit(X_train_selected, y_train)
#model=np.loads(model2)
#Streamlit deployment
def cancer_pred(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float32)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model2.predict(input_data_reshaped)
    return prediction[0]

#selected_features_list = ['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean',
       #'concave points_mean', 'radius_worst', 'perimeter_worst', 'area_worst',
       #'concavity_worst', 'concave points_worst']
selected_features_list = selected_features.columns

# Set the title and description of the web application
st.title("Cancer Prediction")
st.markdown("Enter the input features to predict cancer.")

# Create input fields for the selected features
input_features = []
for feature in selected_features_list:
    value = st.text_input(feature)
    input_features.append(value)

# Make prediction when the button is clicked
if st.button("Predict"):
    prediction = cancer_pred(input_features)
    st.write("Prediction:", prediction)
