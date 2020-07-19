import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16


def func():
    """
    What's VGG: https://zhuanlan.zhihu.com/p/41423739
    """
    # prebuild model with pre-trained weights on imagenet
    model = VGG16(weights='imagenet', include_top=True)
    model.compile(optimizer='sgd', loss='categorical_crossentropy')

    # resize into VGG16 trained images' format
    im = cv2.resize(cv2.imread('../../resource/img/dog.jpeg'), (224, 224)).astype(np.float32)
    im = np.expand_dims(im, axis=0)

    # https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    out = model.predict(im)
    index = np.argmax(out)
    print(index)
    plt.plot(out.ravel())
    plt.show()
    # this should print 820 for steaming train


if __name__ == "__main__":
    func()
    sys.exit()
