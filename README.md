# ML

## Environment
https://docs.floydhub.com/guides/environments/
```shell script
pip install --upgrade tensorflow==2.2.0
pip install --upgrade Keras==2.3.1
```

## Test Data

### MNIST (11M)
https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```shell script
mv mnist.npz ~/.keras/datasets
```

### Cifar (170M)
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```shell script
mv cifar-10-python.tar.gz ~/.keras/datasets
cd ~/.keras/datasets
mv cifar-10-python.tar.gz cifar-10-batches-py.tar.gz
```

### VGG16 Model (500M)
https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
```shell script
mv vgg16_weights_tf_dim_ordering_tf_kernels.h5 ~/.keras/models
```
