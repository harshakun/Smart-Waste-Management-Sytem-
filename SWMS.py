#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# Set the path to your dataset folder
dataset_folder = "C:\\Users\\guded\\Downloads\\archive (4)\\garbage_classification"

# Create a list to store file paths and corresponding labels
data = []

# Iterate through each category in the dataset folder
for category in os.listdir(dataset_folder):
    category_path = os.path.join(dataset_folder, category)
    
    # Check if it's a directory
    if os.path.isdir(category_path):
        # Get a list of all file names in the category
        file_names = os.listdir(category_path)
        
        # Add (file path, label) pairs to the data list
        data += [(os.path.join(category_path, file), category) for file in file_names]

# Split the data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Convert the train set to a DataFrame and save it as a CSV file
train_df = pd.DataFrame(train_data, columns=["FilePath", "Label"])
train_csv_path = "C:\\Users\\guded\\Downloads\\train_dataset.csv"
train_df.to_csv(train_csv_path, index=False)

# Convert the test set to a DataFrame and save it as a CSV file
test_df = pd.DataFrame(test_data, columns=["FilePath", "Label"])
test_csv_path = "C:\\Users\\guded\\Downloads\\test_dataset.csv"
test_df.to_csv(test_csv_path, index=False)

print("Dataset split and CSV files created successfully.")


# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


# In[3]:


# Load training data
train_data = pd.read_csv("C:\\Users\\guded\\Downloads\\train_dataset.csv")

# Load testing data
test_data = pd.read_csv("C:\\Users\\guded\\Downloads\\test_dataset.csv")
print("done")


# In[4]:


# Preprocess training data
train_datagen = ImageDataGenerator(rescale=1./255)
train_data_preprocessed = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    x_col='FilePath',
    y_col='Label',
    target_size=(224, 224),
    seed=42,
    batch_size=32,
    class_mode='categorical'
)

# Preprocess testing data
test_datagen = ImageDataGenerator(rescale=1./255)
test_data_preprocessed = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    x_col='FilePath',
    y_col='Label',
    target_size=(224, 224),
    seed=42,
    batch_size=32,
    class_mode='categorical'
)


# In[5]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# In[6]:


train_data_preprocessed = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    x_col='FilePath',
    y_col='Label',
    target_size=(224, 224),
    seed=42,
    batch_size=32,
    class_mode='categorical'
)


# In[7]:


# Create a new model with modified architecture
model = Sequential([
    MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),  # Increased number of neurons
    Dense(12, activation='softmax')
])


# In[8]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[10]:


# Train the model with increased epochs
history = model.fit(
    train_data_preprocessed,
    steps_per_epoch=len(train_data)//32,
    epochs=30,  # Increased number of epochs
    validation_data=test_data_preprocessed,
    validation_steps=len(test_data)//32
)


# In[11]:


# Evaluate the model
_, accuracy = model.evaluate(test_data_preprocessed)
print(f'Accuracy: {accuracy}')


# In[12]:


import tensorflow as tf


# In[13]:


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


# In[14]:


with open('waste_management.tflite', 'wb') as f:
    f.write(tflite_model)


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Predictions
predictions = model.predict(test_data_preprocessed)
predicted_classes = np.argmax(predictions, axis=1)

# True labels
true_labels = test_data_preprocessed.classes

# Accuracy graph
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[16]:


# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=train_data_preprocessed.class_indices.keys(), yticklabels=train_data_preprocessed.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification report
class_report = classification_report(true_labels, predicted_classes, target_names=train_data_preprocessed.class_indices.keys())
print("Classification Report:\n", class_report)


# In[20]:


# Accuracy score
accuracy = accuracy_score(true_labels, predicted_classes)
print("Accuracy Score:", accuracy)


# In[18]:


pip install matplotlib


# In[19]:


import matplotlib.pyplot as plt

# ... (Previous code for model training)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()


# In[ ]:




