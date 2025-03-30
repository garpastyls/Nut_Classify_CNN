import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam

# Configuration parameters
IMG_WIDTH, IMG_HEIGHT = 150, 150  # Image dimensions
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)  # Input shape for the model
EPOCHS = 10  # Number of training epochs
BATCH_SIZE = 10  # Batch size for training
TRAIN_SAMPLES = 4000  # Number of training samples
VALIDATION_SAMPLES = 500  # Number of validation samples
TEST_SAMPLES = 500  # Number of test samples
DATA_DIR = "./data"  # Root directory for data
TRAIN_DIR = os.path.join(DATA_DIR, "train")  # Training dataset path
VAL_DIR = os.path.join(DATA_DIR, "val")  # Validation dataset path
TEST_DIR = os.path.join(DATA_DIR, "test")  # Test dataset path

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), input_shape=INPUT_SHAPE, activation='relu'),  # First convolutional layer
    MaxPooling2D(pool_size=(2, 2)),  # First max pooling layer
    Conv2D(32, (3, 3), activation='relu'),  # Second convolutional layer
    MaxPooling2D(pool_size=(2, 2)),  # Second max pooling layer
    Conv2D(64, (3, 3), activation='relu'),  # Third convolutional layer
    MaxPooling2D(pool_size=(2, 2)),  # Third max pooling layer
    Flatten(),  # Flatten layer to convert feature maps into a vector
    Dense(64, activation='relu'),  # Fully connected layer with 64 neurons
    Dropout(0.5),  # Dropout layer to prevent overfitting
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Data generators for training, validation, and testing

datagen = ImageDataGenerator(rescale=1.0 / 255)  # Normalize pixel values

train_generator = datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='binary')

val_generator = datagen.flow_from_directory(
    VAL_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='binary')

test_generator = datagen.flow_from_directory(
    TEST_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='binary')

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=TRAIN_SAMPLES // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=VALIDATION_SAMPLES // BATCH_SIZE
)

# Evaluate the model on test data
scores = model.evaluate(test_generator, steps=TEST_SAMPLES // BATCH_SIZE)
print(f"Test accuracy: {scores[1] * 100:.2f}%")

# Save the trained model
model.save("nut_classifier.h5")  # Save model in .h5 format