# Fake-News-Detection-using-ML-models
Fake news detection and evaluation of various Machine learning models.
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the true news dataset
true_data = pd.read_csv(r'Truedata.csv')
true_data['label'] = 1

# Load the fake news dataset
fake_data = pd.read_csv(r'Fakedata.csv')
fake_data['label'] = 0

# Concatenate the datasets
data = pd.concat([true_data, fake_data], ignore_index=True)
data = data.sample(frac=1, random_state=42)
data

# Preprocess the data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
model = SVC(kernel='linear', C=1, gamma='auto')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

new_truedata = pd.read_csv(r'newdata_True.csv')
new_fakedata = pd.read_csv(r'newdata_Fake.csv')

# Combine the two dataframes into a single dataframe
test_data = pd.concat([new_truedata, new_fakedata], ignore_index=True)

# Shuffle the data to ensure randomness
test_data = test_data.sample(frac=1, random_state=42)

test_data


# Preprocess the testing data
test_X = vectorizer.transform(test_data['text'])


# Predict the labels of the testing data using the trained SVM model
test_y_pred = model.predict(test_X)

test_y_pred

#### Labelling test data with predicted lable's
test_data['label'] = test_y_pred
test_data

#### Labelling test data with actual lables
new_truedata = pd.read_csv(r'C:\Users\khazi\Desktop\archive\News _dataset\newdata_True.csv')
new_fakedata = pd.read_csv(r'C:\Users\khazi\Desktop\archive\News _dataset\newdata_Fake.csv')

new_truedata['label'] = 1
new_fakedata['label'] = 0
# Combine the two dataframes into a single dataframe
test_data1 = pd.concat([new_truedata, new_fakedata], ignore_index=True)

# Shuffle the data to ensure randomness
test_data1 = test_data.sample(frac=1, random_state=42)

test_data1

test_y = test_data1['label']

#### Accuracy and different parameters of Test predicted and labelled data

from sklearn import metrics

# Load the true labels for the new data (if available)
# This assumes that the true labels are stored in a column called 'label'


# Calculate the accuracy of the predictions
SVM_accuracy = metrics.accuracy_score(test_y, test_y_pred)

# Calculate the precision, recall, and F1 score of the predictions
precision = metrics.precision_score(test_y, test_y_pred)
recall = metrics.recall_score(test_y, test_y_pred)
f1_score = metrics.f1_score(test_y, test_y_pred)

# Print the evaluation metrics
print('SVM_accuracy:', SVM_accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1_score)

#### Calculation of consfusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calculate the confusion matrix
conf_matrix = confusion_matrix(test_y, test_y_pred)

# Visualize the confusion matrix using a heatmap
sns.heatmap(conf_matrix, annot=True, fmt='g')
from sklearn.metrics import classification_report

# Calculate the precision, recall, and F1-score
report = classification_report(test_y, test_y_pred)
print(report)
