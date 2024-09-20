#!/usr/bin/python3

import os
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore

# Define paths
dataset_path = r'/home/frost/Projects/Traffic_Management_System/Machine Learning Model/Testing_Dataset'

# Step 1: Prepare the Data
# Separate generators for training and validation to avoid augmenting validation images
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True, 
                                   validation_split=0.2)  # 80% train, 20% validation

# Validation data should not have augmentation
valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load training and validation data
train_set = train_datagen.flow_from_directory(dataset_path,
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode='binary',  # Binary classification (vehicle vs non-vehicle)
                                              subset='training')    # Use 80% for training

validation_set = valid_datagen.flow_from_directory(dataset_path,
                                                   target_size=(64, 64),
                                                   batch_size=32,
                                                   class_mode='binary',
                                                   subset='validation')  # Use 20% for validation

# Calculate steps per epoch to avoid running out of data
steps_per_epoch = len(train_set)
validation_steps = len(validation_set)

# Step 2: Build the CNN Model
model = Sequential()

# Convolutional Layer 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully Connected Layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(1, activation='sigmoid'))  # Binary classification (vehicle vs non-vehicle)

# Step 3: Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Step 4: Train the Model
# Add EarlyStopping and ModelCheckpoint for better training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('vehicle_detector_best.keras', monitor='val_loss', save_best_only=True)

# Add .repeat() to ensure data generator doesn't run out of data
model.fit(train_set.__iter__(),
          steps_per_epoch=steps_per_epoch,
          epochs=25,
          validation_data=validation_set.__iter__(),
          validation_steps=validation_steps,
          callbacks=[early_stopping, checkpoint])

# Step 5: Save the Trained Model
model_save_path = r'/home/frost/Projects/Traffic_Management_System/Machine Learning Model/Testing_Dataset/vehicle_detector.h5'
model.save(model_save_path)
print(f'Model saved at {model_save_path}')
