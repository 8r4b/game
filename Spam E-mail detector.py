# Importing dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and preprocess the dataset
df = pd.read_csv(r"C:\Users\msi-pc\Downloads\mail_data.csv")

# Convert null values to empty strings
df = df.where(pd.notnull(df), '')

# Convert the 'Category' column into binary format: 'ham' -> 0, 'spam' -> 1
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

# Split the dataset into training and testing sets (80% train, 20% test)
x = df['Message']
y = df['Category'].astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert the message text into numerical format using TF-IDF
vec = TfidfVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increased max_iter to ensure convergence
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Calculate accuracy of the model
accuracy = acc(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Testing with a sample spam message
sample = [
    "Subject: Congratulations! You've Won a $1000 Gift Card! "
    "Dear Valued Customer, We are thrilled to inform you that you have been selected as the lucky winner of our exclusive $1000 gift card giveaway! "
    "This is our way of saying thank you for being a loyal customer. "
    "To claim your prize, simply click the link below and fill out the required information: Claim Your Prize Now! "
    "Don't miss out on this amazing opportunity! Act fast, as this offer is only available for a limited time. "
    "Best regards, The Promotions Team. This is a promotional message. If you wish to unsubscribe, please click here."
]

# Transform the sample message into the same numerical format
sample = vec.transform(sample)

# Make prediction on the sample message
pred = model.predict(sample)

# Output the prediction result
if pred == 1:
    print("Spam")
else:
    print("Ham")