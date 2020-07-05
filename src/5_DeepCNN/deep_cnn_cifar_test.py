import sys

import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD


def to_image_array(file_name):
    dir_path = '../../resource/img/'
    image = Image.open(dir_path + file_name).resize((32, 32)).convert('RGB')
    image_data = np.array(image, dtype=np.float32)
    return image_data


def func():
    # load model
    model_architecture = 'cifar10_architecture.json'
    model_weights = 'cifar10_weights.h5'
    model = model_from_json(open(model_architecture).read())
    model.load_weights(model_weights)

    # load images
    img_names = ['cat.jpeg', 's_cat.jpg', 'dog.jpeg', 'deer.jpg']
    imgs = [to_image_array(img_name) for img_name in img_names]
    imgs = np.array(imgs) / 255

    model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

    # predict
    # 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
    # 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
    predictions = np.argmax(model.predict(imgs), axis=-1)
    print(predictions)


if __name__ == "__main__":
    func()
    sys.exit()
