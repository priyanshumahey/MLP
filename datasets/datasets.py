from torchvision import datasets

def numbers_dataset(location='./data', download=True):
    train_set = datasets.MNIST(location, train=True, download=download)
    test_set = datasets.MNIST(location, train=False, download=download)

    train_set_array = train_set.data.numpy()
    test_set_array = test_set.data.numpy()
    return train_set_array, test_set_array