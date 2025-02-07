import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle  # For saving the tokenizer

# Load the dataset
with open(r"C:\Users\msi-pc\Downloads\archive (3)\shakespeare.txt", 'r', encoding='utf-8') as file:
    data = file.read()

# Tokenize the data with a vocabulary limit
max_vocab_size = 1000  # Limit vocabulary size for efficient training
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts([data])
total_words = min(len(tokenizer.word_index) + 1, max_vocab_size)

# Create input sequences
input_sequences = []
for line in data.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Create padding
max_sequence_length = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Create features (X) and labels (y)
x, y = input_sequences[:, :-1], input_sequences[:, -1]

# Build the model
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=50, input_length=max_sequence_length - 1))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(x, y, epochs=5, verbose=1)  # Train for a few epochs

# Function to generate text
def generate_text(seed_text, next_words, max_sequence_length):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)[0]
        output_word = tokenizer.index_word.get(predicted_word_index, "")
        seed_text += " " + output_word
    return seed_text

# Example usage
print(generate_text("To be or not to be", 5, max_sequence_length))  # Generate 5 words

# Save the model
model.save('shakespeare_model.h5')  # Save the model to a file
print("Model saved as 'shakespeare_model.h5'")

# Save the tokenizer for later use
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer saved as 'tokenizer.pkl'")