from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def numbers_dataset(location='./data', download=True, normalize=False, Torch=False):
    train_set = datasets.MNIST(location, train=True, download=download)
    test_set = datasets.MNIST(location, train=False, download=download)
    
    if Torch:
        train_data = datasets.MNIST(location, train=True, download=download, transform=transforms.ToTensor())
        test_data = datasets.MNIST(location, train=False, download=download, transform=transforms.ToTensor())
        return train_data, test_data

    if normalize:
        train_set_array = train_set.data.numpy()/255
        train_set_labels = train_set.targets.numpy()/255
        test_set_array = test_set.data.numpy()/255
        test_set_labels = test_set.targets.numpy()/255

    elif not normalize:
        train_set_array = train_set.data.numpy()
        train_set_labels = train_set.targets.numpy()
        test_set_array = test_set.data.numpy()
        test_set_labels = test_set.targets.numpy()

    return train_set_array, test_set_array, train_set_labels, test_set_labels


def visualize(dataset, label, image_num):
    mnist_image = dataset[image_num].reshape(28, 28)
    plt.imshow(mnist_image)
    plt.title(f'True Value: {label[image_num]}')
    plt.show()