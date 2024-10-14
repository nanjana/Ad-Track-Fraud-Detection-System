# Importing all the necessary libraries 

import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Read the dataset from the csv file
file_path = 'sampled_train_dataset.csv'
# Assigning the Dataframe
df = pd.read_csv(file_path)

# Read the dataset from the csv file
file_path_test = 'sampled_test_dataset.csv'
# Assigning the Dataframe
df_test = pd.read_csv(file_path_test)
#Data-Preprocessing

missing_values = df.isnull().sum()

# Display columns with missing values
print("Columns with missing values:")
print(missing_values[missing_values > 0])

# Display unique values in each column
def fraction_unique(x):
    return len(df[x].unique())

number_unique_vals = {x : fraction_unique(x) for x in df.columns}
print(number_unique_vals)

#Exploratory Data Analysis

#Plot App column distribution
app_column = df['app']

# Plot the distribution using Matplotlib
plt.figure(figsize=(10, 6))
plt.hist(app_column, bins=20, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
plt.title('Distribution of App Column')
plt.xlabel('App Values')
plt.ylabel('Count')
plt.show()

#Plot Device column distribution
app_column = df['device']

# Plot the distribution using Matplotlib
plt.figure(figsize=(10, 6))
plt.hist(app_column, bins=20, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
plt.title('Distribution of Device Column')
plt.xlabel('Device Values')
plt.ylabel('Count')
plt.show()

#Plot App column distribution
app_column = df['channel']

# Plot the distribution using Matplotlib
plt.figure(figsize=(10, 6))
plt.hist(app_column, bins=20, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
plt.title('Distribution of Channel Column')
plt.xlabel('Channel Values')
plt.ylabel('Count')
plt.show()

#Plot App column distribution
app_column = df['os']

# Plot the distribution using Matplotlib
plt.figure(figsize=(10, 6))
plt.hist(app_column, bins=20, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
plt.title('Distribution of os Column')
plt.xlabel('os Values')
plt.ylabel('Count')
plt.show()

mean = (df.is_attributed.values == 1).mean()

# Set the color palette
pal = ['skyblue', 'salmon']

# Plot the distribution using Seaborn
plt.figure(figsize=(8, 8))
sns.set(font_scale=1.2)
ax = sns.barplot(x=['Fraudulent (1)', 'Not Fraudulent (0)'], y=[mean, 1 - mean], palette=pal)
ax.set(xlabel='Target Value', ylabel='Probability', title='Target value distribution')

# Annotate the bars with percentages
for p, uniq in zip(ax.patches, [mean, 1 - mean]):
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 0.01,
            '{}%'.format(round(uniq * 100, 2)),
            ha="center")

plt.show()

#Feature Engineering

#for train dataset
#Deriving some new features from the existing ones. Extracting new features from 'click_time' column
def timeFeatures(df):
    # Derive new features using the click_time column
    df['datetime'] = pd.to_datetime(df['click_time'])
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    return df

# Creating new datetime variables and eliminating old ones
train_sample = timeFeatures(df)
train_sample.drop(['click_time', 'datetime'], axis=1, inplace=True)
print(train_sample.head())

#performing IP grouping based on features
ip_count = df.groupby('ip').size().reset_index(name='ip_count').astype('int16')
print(ip_count.head())

def grouped_features(df):
    # Count of occurrences of each IP address
    ip_count = df.groupby('ip').size().reset_index(name='ip_count').astype('uint16')

    # Count of occurrences for each combination of IP, day_of_week, and hour
    ip_day_hour = df.groupby(['ip', 'day_of_week', 'hour']).size().reset_index(name='ip_day_hour').astype('uint16')

    # Count of occurrences for each combination of IP, hour, and channel
    ip_hour_channel = df[['ip', 'hour', 'channel']].groupby(['ip', 'hour', 'channel']).size().reset_index(name='ip_hour_channel').astype('uint16')

    # Count of occurrences for each combination of IP, hour, and os
    ip_hour_os = df.groupby(['ip', 'hour', 'os']).channel.count().reset_index(name='ip_hour_os').astype('uint16')

    # Count of occurrences for each combination of IP, hour, and app
    ip_hour_app = df.groupby(['ip', 'hour', 'app']).channel.count().reset_index(name='ip_hour_app').astype('uint16')

    # Count of occurrences for each combination of IP, hour, and device
    ip_hour_device = df.groupby(['ip', 'hour', 'device']).channel.count().reset_index(name='ip_hour_device').astype('uint16')

    # Merge the new aggregated features with the original DataFrame
    df = pd.merge(df, ip_count, on='ip', how='left')
    del ip_count
    df = pd.merge(df, ip_day_hour, on=['ip', 'day_of_week', 'hour'], how='left')
    del ip_day_hour
    df = pd.merge(df, ip_hour_channel, on=['ip', 'hour', 'channel'], how='left')
    del ip_hour_channel
    df = pd.merge(df, ip_hour_os, on=['ip', 'hour', 'os'], how='left')
    del ip_hour_os
    df = pd.merge(df, ip_hour_app, on=['ip', 'hour', 'app'], how='left')
    del ip_hour_app
    df = pd.merge(df, ip_hour_device, on=['ip', 'hour', 'device'], how='left')
    del ip_hour_device
    
    return df

df = grouped_features(df)
print(df.head())

#for test dataset
#Deriving some new features from the existing ones. Extracting new features from 'click_time' column
def timeFeatures_test(df_test):
    # Derive new features using the click_time column
    df_test['datetime'] = pd.to_datetime(df_test['click_time'])
    df_test['day_of_week'] = df_test['datetime'].dt.dayofweek
    df_test['day_of_year'] = df_test['datetime'].dt.dayofyear
    df_test['month'] = df_test['datetime'].dt.month
    df_test['hour'] = df_test['datetime'].dt.hour
    return df_test

# Creating new datetime variables and eliminating old ones
train_sample_test = timeFeatures_test(df_test)
train_sample_test.drop(['click_time', 'datetime'], axis=1, inplace=True)
print(train_sample_test.head())

#performing IP grouping based on features
ip_count_test = df_test.groupby('ip').size().reset_index(name='ip_count').astype('int16')
print(ip_count_test.head())

def grouped_features_test(df_test):
    # Count of occurrences of each IP address
    ip_count_test = df_test.groupby('ip').size().reset_index(name='ip_count').astype('uint16')

    # Count of occurrences for each combination of IP, day_of_week, and hour
    ip_day_hour = df_test.groupby(['ip', 'day_of_week', 'hour']).size().reset_index(name='ip_day_hour').astype('uint16')

    # Count of occurrences for each combination of IP, hour, and channel
    ip_hour_channel = df_test[['ip', 'hour', 'channel']].groupby(['ip', 'hour', 'channel']).size().reset_index(name='ip_hour_channel').astype('uint16')

    # Count of occurrences for each combination of IP, hour, and os
    ip_hour_os = df_test.groupby(['ip', 'hour', 'os']).channel.count().reset_index(name='ip_hour_os').astype('uint16')

    # Count of occurrences for each combination of IP, hour, and app
    ip_hour_app = df_test.groupby(['ip', 'hour', 'app']).channel.count().reset_index(name='ip_hour_app').astype('uint16')

    # Count of occurrences for each combination of IP, hour, and device
    ip_hour_device = df_test.groupby(['ip', 'hour', 'device']).channel.count().reset_index(name='ip_hour_device').astype('uint16')

    # Merge the new aggregated features with the original DataFrame
    df_test = pd.merge(df_test, ip_count_test, on='ip', how='left')
    del ip_count_test
    df_test = pd.merge(df_test, ip_day_hour, on=['ip', 'day_of_week', 'hour'], how='left')
    del ip_day_hour
    df_test = pd.merge(df_test, ip_hour_channel, on=['ip', 'hour', 'channel'], how='left')
    del ip_hour_channel
    df_test = pd.merge(df_test, ip_hour_os, on=['ip', 'hour', 'os'], how='left')
    del ip_hour_os
    df_test = pd.merge(df_test, ip_hour_app, on=['ip', 'hour', 'app'], how='left')
    del ip_hour_app
    df_test = pd.merge(df_test, ip_hour_device, on=['ip', 'hour', 'device'], how='left')
    del ip_hour_device
    
    return df_test

df_test = grouped_features_test(df_test)
print(df_test.head())

#Model Training

# Selecting independent variables
X = train_sample[['app', 'device', 'os', 'channel', 'day_of_week', 'month', 'hour', 'ip', 'day_of_year']]

# Dependent variable
y = train_sample['is_attributed']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Logistic Regression

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_report_str}')

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Plot Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.plot(recall, precision, label=f'AUC = {auc(recall, precision):.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# Plot Feature Importance
if hasattr(model, 'coef_'):
    feature_importance = pd.Series(model.coef_[0], index=X.columns)
    feature_importance = feature_importance / feature_importance.abs().sum()  # Normalize
    feature_importance = feature_importance.sort_values()
    feature_importance.plot(kind='barh', color='teal')
    plt.title('Feature Importance')
    plt.xlabel('Coefficient Magnitude')
    plt.show()

#Decision Trees

# Initialize the Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_report_str}')

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=X.columns.tolist(), class_names=['Not Attributed', 'Attributed'], filled=True, rounded=True)
plt.title('Decision Tree')
plt.show()

#Random Forest

# Initialize the Random Forest model
model = RandomForestClassifier(random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_report_str}')

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance = feature_importance / feature_importance.sum()  # Normalize
feature_importance = feature_importance.sort_values()
feature_importance.plot(kind='barh', color='teal')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.show()

#Naive-Bayes
# Initialize the Naive Bayes model
model = GaussianNB()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_report_str}')

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#K-Nearest Neighbor

# Initialize the KNN model
model = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors (n_neighbors) based on your preference

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_report_str}')

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

error_rate = []

# Vary the number of neighbors from 1 to 20
for i in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error_rate.append(1 - accuracy_score(y_test, y_pred))

# Plotting the error rate
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.show()

#Support Vector Machine
# Standardize the features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the SVM model
model = SVC(kernel='linear', random_state=42)  # You can choose different kernels like 'linear', 'rbf', 'poly', etc.

# Train the model on the training set
model.fit(X_train_scaled, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_report_str}')

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Model Evaluation using Test dataset

# Assuming 'X_test' is your test features
X_test = train_sample_test[['app', 'device', 'os', 'channel', 'day_of_week', 'month', 'hour', 'ip', 'day_of_year']]

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)  # You can adjust the value as needed
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

#print predictions
print(lr_pred)
print(dt_pred)
print(rf_pred)
print(nb_pred)
print(knn_pred)

#Model Comparison
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(probability=True)  # Enable probability estimates for ROC-AUC
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc
    }

# Convert results to a DataFrame for easy visualization
results_df = pd.DataFrame(results).T

# Display results
print("Model Comparison:")
print(results_df)

# Visualize the results
plt.figure(figsize=(12, 8))
sns.barplot(x=results_df.index, y='Accuracy', data=results_df, palette='viridis')
plt.title('Accuracy Comparison')
plt.show()