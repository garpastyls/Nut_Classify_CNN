# Nut Classification CNN

## Project Description
This project is a Convolutional Neural Network (CNN) designed to classify nuts as "good" or "bad." The neural network is trained on images of nuts and can then predict the quality of new images.

The model was developed as a diploma project and can be adapted for classifying other products or industrial tasks (e.g., quality control in manufacturing or object recognition in customs inspection).

The model achieved **94.80% accuracy** on a test dataset of 5000 images.

## Technologies Used
- Python 3.8+
- TensorFlow + Keras
- NumPy
- ImageDataGenerator for image loading

## Installation
Before starting, install the required dependencies:
```sh
pip install tensorflow numpy keras
```

## Data Preparation
A total of **5000 images of nuts** were taken on a **green background**, divided into two classes:
- **2500 images of good nuts**
- **2500 images of bad nuts**

Example images (good nut/bad nut):
- ![Good Nut](https://github.com/garpastyls/Nut_Classify_CNN/blob/main/Good%20nut.png)
- ![Bad Nut](https://github.com/garpastyls/Nut_Classify_CNN/blob/main/Bad%20nut.png)

### Data Organization
Create a `data/` directory with the following subdirectories:
```
data/
├── train/      # 4000 images (80%)
│   ├── good/   # 2000 images
│   ├── bad/    # 2000 images
├── val/        # 500 images (10%)
│   ├── good/   # 250 images
│   ├── bad/    # 250 images
├── test/       # 500 images (10%)
│   ├── good/   # 250 images
│   ├── bad/    # 250 images
```

## Training the Model
Run the Python script to train the model:
```sh
python Nut_CNN.py
```
After training, the model will be saved in the file `nut_classifier.h5`. The **.h5** file format is used to store trained model weights in Keras.

## Example Usage
```python
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = load_model("nut_classifier.h5")

# Load an image
img = image.load_img("path/to/image.jpg", target_size=(150, 150))

# Convert image to an array and normalize
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# Predict the class of the image
prediction = model.predict(x)[0][0]

# Display the result
print("Good nut" if prediction > 0.5 else "Bad nut")
```

## How to Train the Model on Your Own Data?
1. Prepare your dataset by organizing images into `train/`, `val/`, and `test/` folders.
2. Update the script with your dataset paths.
3. Run the training process.
4. Use the trained model for predictions on new images.

## Possible Improvements
- You can add hyperparameter configuration via a config file.
- Improve data augmentation for better generalization.
- Adding more layers requires testing for potential accuracy improvement or degradation.

## License
This project is licensed under the **MIT** license.
