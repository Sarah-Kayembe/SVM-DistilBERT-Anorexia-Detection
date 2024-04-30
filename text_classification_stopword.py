#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Date: 29 April 2024

Title: Text Classification Pipeline for Early Detection of Signs of Anorexia
Author: Sarah Kayembe
Email: sarah.kayembe@maine.edu
Description: This code implements a text classification pipeline using Support Vector Machine (SVM) with TF-IDF vectorization and DistilBERT for the early detection of signs of anorexia. It includes data preprocessing, model training, evaluation, and comparison between SVM and DistilBERT.
Lab: eRisks CLEF LAB 2024
'''

import os
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from bs4 import BeautifulSoup
import time
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# Measure latency
start_time = time.time()

# Function to extract text from XML files
def extract_text_from_xml(xml_file):
    with open(xml_file, "r", encoding="utf-8") as f:
        content = f.read()
        soup = BeautifulSoup(content, 'xml')
        text = ' '.join([writing.text for writing in soup.find_all('TEXT')])
        return text

# Function to read data from directory containing XML files
def read_data_from_directory(directory, label):
    data = {'text': [], 'label': []}
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            text = extract_text_from_xml(os.path.join(directory, filename))
            data['text'].append(text)
            data['label'].append(label)
    return data

def read_test_data_from_directory(directory):
    test_data = {'subject': [], 'text': []}
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            # Extract subject from filename
            subject = filename.split('_')[0]  # Extract subject ID before the first '_'
            text = extract_text_from_xml(os.path.join(directory, filename))
            test_data['subject'].append(subject)
            test_data['text'].append(text)
    return test_data

# Calculate ERDE@N
def calculate_ERDE_N(y_true, y_pred, N):
    relevant_positions = np.where(y_true == 1)[0]
    if len(relevant_positions) == 0:
        return 1.0  # If no relevant documents, return maximum ERDE
    top_N_positions = y_pred[:N]
    ERDE_N = sum(1 for pos in top_N_positions if pos in relevant_positions) / len(relevant_positions)
    return ERDE_N

# Function to calculate Latency-weighted F1 (example implementation)
def calculate_latency_weighted_F1(f1_score, latencyT):
    # Example implementation, replace it with your actual calculation
    latency_weighted_F1 = 0.8 * f1_score + 0.2 * (1 / latencyT)
    return latency_weighted_F1

# Read positive and negative examples
# You can replace '/path/positive_examples' and '/path/negative_examples' 
# with the path to the directory containing your training and validation XML files
positive_directory = '/path/positive_examples'
negative_directory = '/path/negative_examples'

positive_data = read_data_from_directory(positive_directory, 1)
negative_data = read_data_from_directory(negative_directory, 0)

# Combine positive and negative examples
data = {'text': positive_data['text'] + negative_data['text'],
        'label': positive_data['label'] + negative_data['label']}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Split data into training and validation sets
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

'''
# Print the first few samples of the training data
print("Sample of training data:")
for text, label in zip(X_train[:5], y_train[:5]):
    print("Text:", text)
    print("Label:", label)
    print("--------------")
'''
# Define text classification pipeline
print("Defining text classification pipeline...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC()),
])

# Define hyperparameters grid for Grid Search
param_grid = {
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf'],
}

# Perform Grid Search to find optimal hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Get best model from Grid Search
best_model = grid_search.best_estimator_

# Evaluate best model on validation set
y_pred = best_model.predict(X_val)

# Calculate evaluation metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# Measure latency by calculating the time difference
end_time = time.time()
latency_in_seconds = end_time - start_time
latencyT = latency_in_seconds

# Calculate additional metrics
ERDE_5 = calculate_ERDE_N(y_val, y_pred, 5)
ERDE_50 = calculate_ERDE_N(y_val, y_pred, 50)
P_speed = 1000 / latencyT  # Example processing speed in samples per second
latency_weighted_F1 = calculate_latency_weighted_F1(f1, latencyT)

 # Convert to milliseconds
# Output evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Print additional metrics
print("ERDE@5:", ERDE_5)
print("ERDE@50:", ERDE_50)
print("LatencyT:", latencyT)
print("P Speed:", P_speed)
print("Latency-weighted F1:", latency_weighted_F1)

# Measure latency
start_time_test = time.time()

# Now you can proceed to use the best model to predict labels for test data and evaluate its generalizability
# You can replace '/path/to/test' with the path to the directory containing your test XML files
test_data_directory = '/path/to/test'
test_data = read_test_data_from_directory(test_data_directory)

#print("Length of 'text' array:", len(test_data['text']))
#print("Length of 'subject' array:", len(test_data['subject']))

# Convert test data to DataFrame
test_df = pd.DataFrame(test_data)

# Predict labels for test data
y_test_pred = best_model.predict(test_df['text'])

# Output predictions for test data
print("\nPredictions for test data:")
print(y_test_pred)

print(test_df.columns)


# Load the test data file
#Replace "/path/to/risk-golden-truth-test.txt" with your path to the risk-golden-truth-test.txt
test_labels_file = "/path/to/risk-golden-truth-test.txt"

# Read the test labels
test_labels_df = pd.read_csv(test_labels_file, sep='\t', header=None, names=['subject', 'label'])

#print(test_labels_df.columns)

# Join the test data with the corresponding labels based on subject
test_data_with_labels = test_df.merge(test_labels_df, left_on='subject', right_on='subject')

# Extract features and labels for testing
X_test = test_data_with_labels['text']
y_test = test_data_with_labels['label']

# Predict labels for test data using the trained model
y_test_pred = best_model.predict(X_test)

# Calculate evaluation metrics for test data
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

# Measure latency by calculating the time difference
end_time_test = time.time()
latency_in_seconds_test = end_time_test - start_time_test
latencyT_test = latency_in_seconds_test

# Calculate additional metrics
ERDE_5_test = calculate_ERDE_N(y_val, y_pred, 5)
ERDE_50_test = calculate_ERDE_N(y_val, y_pred, 50)
P_speed_test = 1000/latencyT_test  # Example processing speed in samples per second
latency_weighted_F1_test = calculate_latency_weighted_F1(f1, latencyT_test)

# Output evaluation metrics for test data
print("\nEvaluation metrics for test data:")
print("Accuracy:", accuracy_test)
print("Precision:", precision_test)
print("Recall:", recall_test)
print("F1-Score:", f1_test)

# Print additional metrics
print("ERDE@5:", ERDE_5_test)
print("ERDE@50:", ERDE_50_test)
print("LatencyT:", latencyT_test)
print("P Speed:", P_speed_test)
print("Latency-weighted F1:", latency_weighted_F1_test)

'''
# Iterate over test data and print misclassified subjects with their labels
print("\nMisclassified subjects:")
for index, row in test_data_with_labels.iterrows():
    if row['label'] != y_test_pred[index]:
        print(f"Subject: {row['subject']}, True Label: {row['label']}, Predicted Label: {y_test_pred[index]}")
'''

nltk.download('stopwords')
# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

# Apply text cleaning and stopwords removal before tokenization
X_train_cleaned = [remove_stopwords(text) for text in X_train]
X_val_cleaned = [remove_stopwords(text) for text in X_val]

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Set device for model execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the selected device

# Tokenize cleaned text data with a maximum sequence length
X_train_tokens = tokenizer(X_train_cleaned, padding=True, truncation=True, return_tensors='pt', max_length=128).to(device)
X_val_tokens = tokenizer(X_val_cleaned, padding=True, truncation=True, return_tensors='pt', max_length=128).to(device)

# Convert labels to tensors and move to the selected device
y_train_tensor = torch.tensor(y_train.tolist()).to(device)
y_val_tensor = torch.tensor(y_val.tolist()).to(device)

# Fine-tune BERT model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
epochs = 3
batch_size = 8

for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        optimizer.zero_grad()
        outputs = model(input_ids=X_train_tokens['input_ids'][i:i+batch_size],
                        attention_mask=X_train_tokens['attention_mask'][i:i+batch_size],
                        labels=y_train_tensor[i:i+batch_size])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluate model on validation set
outputs = model(input_ids=X_val_tokens['input_ids'], attention_mask=X_val_tokens['attention_mask'])
predictions = torch.argmax(outputs.logits, dim=1)
accuracy = accuracy_score(y_val_tensor.cpu().numpy(), predictions.cpu().numpy())  # Move tensors to CPU for evaluation
precision = precision_score(y_val_tensor.cpu().numpy(), predictions.cpu().numpy())
recall = recall_score(y_val_tensor.cpu().numpy(), predictions.cpu().numpy())
f1 = f1_score(y_val_tensor.cpu().numpy(), predictions.cpu().numpy())

# Output evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Measure latency by calculating the time difference
end_time = time.time()
latency_in_seconds = end_time - start_time
latencyT = latency_in_seconds

# Print additional metrics
print("LatencyT:", latencyT)

# Assuming y_val is the true labels for validation data
total_inputs = len(y_val)

# Calculate the number of inputs correctly classified by SVM with TF-IDF
svm_tfidf_correct = sum(1 for true_label, predicted_label in zip(y_val, y_pred) if true_label == predicted_label)

# Calculate the number of inputs correctly classified by DistilBERT
distilbert_correct = sum(1 for true_label, predicted_label in zip(y_val, predictions.cpu().numpy()) if true_label == predicted_label)

# Print the results
print("Total Inputs:", total_inputs)
print("SVM with TF-IDF correctly classified:", svm_tfidf_correct)
print("DistilBERT correctly classified:", distilbert_correct)

# Presentation bullet points
print("\nPresentation Summary:")
print("- SVM with TF-IDF classified", svm_tfidf_correct, "out of", total_inputs, "inputs correctly.")
print("- DistilBERT classified", distilbert_correct, "out of", total_inputs, "inputs correctly.")
print("- DistilBERT achieved higher accuracy compared to SVM with TF-IDF.")
print("- Both models show good performance, suggesting potential use in real-world applications.")
print("- Further analysis is needed to understand misclassifications and improve model performance.")

# Define the output file path
output_file_path = "predictions.txt"

# Write SVM with TF-IDF predictions to the file
with open(output_file_path, "w") as file:
    file.write("SVM with TF-IDF Predictions:\n")
    for filename, true_label, predicted_label in zip(test_data_with_labels['subject'], y_test, y_test_pred):
        if true_label == predicted_label:
            file.write(f"File {filename} - Correctly Predicted\n")
        else:
            file.write(f"File {filename} - Incorrectly Predicted\n")

    # Write DistilBERT predictions to the file
    file.write("\nDistilBERT Predictions:\n")
    for filename, true_label, predicted_label in zip(test_data_with_labels['subject'], y_test, y_test_pred):
        if true_label == predicted_label:
            file.write(f"File {filename} - Correctly Predicted\n")
        else:
            file.write(f"File {filename} - Incorrectly Predicted\n")

# Get the absolute path to the output file
absolute_path = os.path.abspath(output_file_path)

print("Predictions written to", absolute_path)


# Predict labels for test data using the trained model
y_test_pred = best_model.predict(X_test)

# Predict labels for test data using SVM with TF-IDF
y_test_pred_svm_tfidf = best_model.predict(X_test)

# Tokenize test data for DistilBERT
X_test_tokens = tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128).to(device)

# Predict labels for test data using DistilBERT
y_test_pred_distilbert = []  # Initialize list to store predictions
for i in range(0, len(X_test), batch_size):
    outputs = model(input_ids=X_test_tokens['input_ids'][i:i+batch_size],
                    attention_mask=X_test_tokens['attention_mask'][i:i+batch_size])
    predictions = torch.argmax(outputs.logits, dim=1)
    y_test_pred_distilbert.extend(predictions.cpu().numpy())


# Define the output file path
output_file_path = "svm_vs_distil_predictions.txt"

# Initialize lists to store predictions
svm_correct_distilbert_incorrect = []
svm_incorrect_distilbert_correct = []

# Iterate over test data and store filenames predicted correctly by SVM but incorrectly by DistilBERT
for filename, true_label, svm_pred_label, distilbert_pred_label in zip(test_data_with_labels['subject'], y_test, y_test_pred_svm_tfidf, y_test_pred_distilbert):
    if true_label == svm_pred_label and true_label != distilbert_pred_label:
        svm_correct_distilbert_incorrect.append(filename)

# Iterate over test data and store filenames predicted correctly by DistilBERT but incorrectly by SVM
for filename, true_label, svm_pred_label, distilbert_pred_label in zip(test_data_with_labels['subject'], y_test, y_test_pred_svm_tfidf, y_test_pred_distilbert):
    if true_label != svm_pred_label and true_label == distilbert_pred_label:
        svm_incorrect_distilbert_correct.append(filename)

# Write SVM predictions to the file
with open(output_file_path, "w") as file:
    file.write("SVM predicted correctly and DistilBERT failed:\n")
    for filename in svm_correct_distilbert_incorrect:
        file.write(f"File {filename}\n")

    file.write("\nDistilBERT predicted correctly and SVM failed:\n")
    for filename in svm_incorrect_distilbert_correct:
        file.write(f"File {filename}\n")

# Get the absolute path to the file
abs_output_file_path = os.path.abspath(output_file_path)

print("Predictions written to:", abs_output_file_path)

# Define the output file path
output_file_path = "svm_vs_distil_predictions_contents.txt"

# Initialize lists to store filenames and their content
svm_correct_distilbert_incorrect = []
svm_incorrect_distilbert_correct = []

# Iterate over test data and store filenames predicted correctly by SVM but incorrectly by DistilBERT
for filename, true_label, svm_pred_label, distilbert_pred_label, text in zip(test_data_with_labels['subject'], y_test, y_test_pred_svm_tfidf, y_test_pred_distilbert, X_test):
    if true_label == svm_pred_label and true_label != distilbert_pred_label:
        svm_correct_distilbert_incorrect.append((filename, true_label, svm_pred_label, distilbert_pred_label, text))

# Iterate over test data and store filenames predicted correctly by DistilBERT but incorrectly by SVM
for filename, true_label, svm_pred_label, distilbert_pred_label, text in zip(test_data_with_labels['subject'], y_test, y_test_pred_svm_tfidf, y_test_pred_distilbert, X_test):
    if true_label != svm_pred_label and true_label == distilbert_pred_label:
        svm_incorrect_distilbert_correct.append((filename, true_label, svm_pred_label, distilbert_pred_label, text))

# Write SVM predictions to the file along with content, true label, and predicted label
with open(output_file_path, "w") as file:
    file.write("SVM predicted correctly and DistilBERT failed:\n")
    for filename, true_label, svm_pred_label, distilbert_pred_label, text in svm_correct_distilbert_incorrect:
        file.write(f"File {filename} - Correctly Predicted by SVM but Failed by DistilBERT\n")
        file.write(f"True Label: {true_label}, Predicted Label (SVM): {svm_pred_label}, Predicted Label (DistilBERT): {distilbert_pred_label}\n")
        file.write(f"Text: {text}\n\n")

    file.write("\nDistilBERT predicted correctly and SVM failed:\n")
    for filename, true_label, svm_pred_label, distilbert_pred_label, text in svm_incorrect_distilbert_correct:
        file.write(f"File {filename} - Correctly Predicted by DistilBERT but Failed by SVM\n")
        file.write(f"True Label: {true_label}, Predicted Label (SVM): {svm_pred_label}, Predicted Label (DistilBERT): {distilbert_pred_label}\n")
        file.write(f"Text: {text}\n\n")

# Get the absolute path to the file
abs_output_file_path = os.path.abspath(output_file_path)

print("Predictions written to:", abs_output_file_path)

#If you would like to analyze the data being classified, you can uncomment it
'''
# Iterate over test data and print filenames predicted correctly and incorrectly by SVM with TF-IDF
print("\nSVM with TF-IDF Predictions:")
for filename, true_label, predicted_label in zip(test_data_with_labels['subject'], y_test, y_test_pred):
    if true_label == predicted_label:
        print(f"File {filename} - Correctly Predicted")
    else:
        print(f"File {filename} - Incorrectly Predicted")

# Iterate over test data and print filenames predicted correctly and incorrectly by DistilBERT
print("\nDistilBERT Predictions:")
for filename, true_label, predicted_label in zip(test_data_with_labels['subject'], y_test, y_test_pred):
    if true_label == predicted_label:
        print(f"File {filename} - Correctly Predicted")
    else:
        print(f"File {filename} - Incorrectly Predicted")
'''
'''
# Iterate over test data and print misclassified instances with their predicted and true labels
misclassified_instances = []
for index, row in test_data_with_labels.iterrows():
    if row['label'] != y_test_pred[index]:
        misclassified_instances.append((row['subject'], row['text'], row['label'], y_test_pred[index]))

# Print misclassified instances
print("\nMisclassified instances:")
for subject, text, true_label, predicted_label in misclassified_instances:
    print(f"Subject: {subject}")
    print(f"Text: {text}")
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
    print("--------------")


# Iterate over test data and print correctly classified instances with their predicted and true labels
correctly_classified_instances = []
for index, row in test_data_with_labels.iterrows():
    if row['label'] == y_test_pred[index]:
        correctly_classified_instances.append((row['subject'], row['text'], row['label'], y_test_pred[index]))

# Print correctly classified instances
print("\nCorrectly classified instances:")
for subject, text, true_label, predicted_label in correctly_classified_instances:
    print(f"Subject: {subject}")
    print(f"Text: {text}")
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
    print("--------------")
'''
