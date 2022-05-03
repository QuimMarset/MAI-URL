import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_hand_image(root_path, image_size):
    path = os.path.join(root_path, 'datasets', 'hand_image.png')
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
    image = cv2.resize(image, (image_size[1], image_size[0]))
    return image


def generate_hand_image_data(root_path, image_size=(80, 64)):
    image_luv = load_hand_image(root_path, image_size)
    num_pixels = image_size[0] * image_size[1]
    data = np.zeros((num_pixels, 5))

    for i in range(image_size[0]):
        for j in range(image_size[1]):
            (l, u, v) = image_luv[i, j]
            data[i*image_size[1] + j] = [i, j, l, u, v]
    
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data


if __name__ == '__main__':
    data = generate_hand_image_data('./')
    print(data.shape)
    print(np.mean(data, axis=0))
    print(np.std(data, axis=0))