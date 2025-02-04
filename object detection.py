import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.datasets import cifar10 # type: ignore
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Check the type of the first image in the training set
print(type(x_train[0]))  # Should print <class 'numpy.ndarray'>

# Normalize the pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert the labels into one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # First convolutional layer
    layers.MaxPooling2D((2, 2)),  # First max pooling layer
    layers.Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    layers.MaxPooling2D((2, 2)),  # Second max pooling layer
    layers.Conv2D(64, (3, 3), activation='relu'),  # Third convolutional layer
    layers.Flatten(),  # Flatten the output for the dense layers
    layers.Dense(64, activation='relu'),  # Dense hidden layer
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, 
                    epochs=10, 
                    batch_size=64, 
                    validation_data=(x_test, y_test))



# CIFAR-10 class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']

# Function to load and preprocess a new image
def load_and_preprocess_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(32, 32))  # Resize to match model input
    img_array = image.img_to_array(img)  # Convert to array
    img_array = img_array.astype('float32') / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Path to your new image
img_path = r"C:\Users\msi-pc\Downloads\dog.jpeg"  # Update this to your image path

# Load and preprocess the image
new_image = load_and_preprocess_image(img_path)

# Make predictions
predictions = model.predict(new_image)

# Get the class with the highest probability
predicted_class = np.argmax(predictions, axis=1)

# Display the result
print(f'The predicted class is: {class_labels[predicted_class[0]]}')

# Optionally, display the image
plt.imshow(image.load_img(img_path))
plt.axis('off')
plt.show()
model.save(r'C:\Users\msi-pc\OneDrive\المستندات')