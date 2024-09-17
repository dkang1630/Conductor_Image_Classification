import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

def load_images_and_labels(src_dir):
    images = []
    labels = []
    label_names = ['def_cor', 'only_cor', 'only_def']
    
    for label in label_names:
        category_path = os.path.join(src_dir, label)
        for filename in os.listdir(category_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(category_path, filename)
                image = Image.open(image_path)
                images.append(np.array(image))  # Keep as is
                labels.append(label)
    
    return images, labels

def save_splits(images, labels, train_dir, test_dir, train_indices, test_indices):
    for index in train_indices:
        image = images[index]
        label = labels[index]
        category_dir = os.path.join(train_dir, label)
        os.makedirs(category_dir, exist_ok=True)
        file_path = os.path.join(category_dir, f'image_{index}.jpg')
        Image.fromarray(image).save(file_path)

    for index in test_indices:
        image = images[index]
        label = labels[index]
        category_dir = os.path.join(test_dir, label)
        os.makedirs(category_dir, exist_ok=True)
        file_path = os.path.join(category_dir, f'image_{index}.jpg')
        Image.fromarray(image).save(file_path)

# Directories
src_directory = r'C:\Users\dkang\OneDrive\Documents\Gray Scale Image Code\Conductor_research\Image\GVN\GVN_Img'
train_directory = r'C:\Users\dkang\OneDrive\Documents\Gray Scale Image Code\Conductor_research\Image\GVN\CNN_GVN_Train_Img'
test_directory = r'C:\Users\dkang\OneDrive\Documents\Gray Scale Image Code\Conductor_research\Image\GVN\CNN_GVN_Test_Img'

# Load images and labels
images, labels = load_images_and_labels(src_directory)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, stratify=labels, random_state=123)

# Generate indices for saving
train_indices = range(len(X_train))
test_indices = range(len(X_test))

# Save the split data
save_splits(X_train, y_train, train_directory, test_directory, train_indices, test_indices)

print('Data split and saved successfully!')
