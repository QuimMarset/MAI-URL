import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_image(image_path, image_size=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        image = cv2.resize(image, (image_size[0], image_size[1]))
    return image


def generate_image_data(image_rgb):
    image_size = image_rgb.shape
    num_pixels = image_size[0] * image_size[1]
    data = np.zeros((num_pixels, 5))

    for i in range(image_size[0]):
        for j in range(image_size[1]):
            (l, u, v) = image_rgb[i, j]
            data[i*image_size[1] + j] = [i, j, l, u, v]

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    
    return scaler, normalized_data