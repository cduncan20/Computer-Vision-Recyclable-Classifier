import torchvision
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from random import random
from skimage.util import random_noise


def gaussian_blur(img):
    image = np.array(img)
    rand_num = random()
    if rand_num <= 0.05:
        image = cv2.GaussianBlur(image, (21, 21), 0)
    return image


# Create image with random noise
def noise(img):
    image = np.array(img)
    rand_num = random()
    if rand_num <= 0.05:
        image = random_noise(image)

        # image is originally a float64 data type, so convert to a uint8 to match others
        image = 255 * image / np.amax(image)  # Current values are between 0 and 1. Change to between 0 and 255
        image = image.astype(np.uint8)  # Convert to uint8 data type
    return image


def get_transform(transform_dict, train):
    transform_list = [transforms.Resize(256)]
    if train:
        if transform_dict['horizontal']:
            transform_list.append(transforms.RandomHorizontalFlip())
        if transform_dict['vertical']:
            transform_list.append(transforms.RandomVerticalFlip())
        if transform_dict['rot30']:
            transform_list.append(transforms.RandomRotation(degrees=30))
        if transform_dict['noise']:
            transform_list.append(transforms.Lambda(gaussian_blur))
        if transform_dict['blur']:
            transform_list.append(transforms.Lambda(noise))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def transform_and_split_data(transform_dict, data_directory_path, val_ratio, test_ratio):
    # Define transforms for training, validation, and test data
    train_transform = get_transform(transform_dict, train=True)
    val_test_transform = get_transform(transform_dict, train=False)

    # Perform Transformations on Training & Test Data
    # NOTE: Data has not yet been split, so train_data, valid_data & test_data hold reference to all images
    train_set = datasets.ImageFolder(data_directory_path, transform=train_transform)
    val_set = datasets.ImageFolder(data_directory_path, transform=val_test_transform)
    test_set = datasets.ImageFolder(data_directory_path, transform=val_test_transform)

    # Split data based on test_size ratio defined above
    total_num_images = len(train_set)  # Total number of images
    image_indices = list(range(total_num_images))  # Define an index to each image
    num_val = int(np.floor(val_ratio * total_num_images))  # Number of images assigned to validation data
    num_test = int(np.floor(test_ratio * total_num_images))  # Number of images assigned to testing data
    np.random.shuffle(image_indices)  # Shuffle all image indices so they are in a random order
    train_index = image_indices[num_test + num_val:]  # Assign image indices belonging to training data
    val_index = image_indices[num_test:num_test + num_val]  # Assign image indices belonging to validation data
    test_index = image_indices[:num_test]  # Assign image indices belonging to test data

    print("Number of training images:", len(train_index))
    print("Number of validation images:", len(val_index))
    print("Number of test images:", len(test_index))

    # Define a random sampler to randomly choose indices for training, validation & test data when loading data
    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)
    test_sampler = SubsetRandomSampler(test_index)

    # Assign split data into loaded data set
    batch_qty = 8
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_qty, num_workers=4)
    val_loader = DataLoader(val_set, sampler=val_sampler, batch_size=batch_qty, num_workers=4)
    test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=batch_qty, num_workers=4)

    return train_loader, val_loader, test_loader


def visualize_training_data(loader, class_names):
    # get some random training images
    dataiter = iter(loader)
    images, labels = dataiter.next()

    # print class labels
    L = labels.numpy()
    out_string = ""
    for i in range(len(L)):
        out_string += "%s " % class_names[L[i]]

    print("")
    print("Classes of Images Shown:", out_string)

    # show images
    imshow(torchvision.utils.make_grid(images))


# Show examples of images in loader
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()


# Create List of Classes based on directory subfolders
def get_class_names(data_dir):
    folder_names = []
    for entry_name in os.listdir(data_dir):
        entry_path = os.path.join(data_dir, entry_name)
        if os.path.isdir(entry_path):
            folder_names.append(entry_name)

    folder_names = sorted(folder_names)  # Sort class names alphabetically
    return folder_names


def load_data():
    data_dir = "Images/TRAINING_&_TEST/TRAINING_&_TEST_IMAGES"
    class_names = get_class_names(data_dir)

    transform_dict = dict([('horizontal', True),
                           ('vertical', True),
                           ('rot30', True),
                           ('noise', True),
                           ('blur', True)])
    val_ratio = 0.2
    test_ratio = 0.2

    # Load data
    print("Loading database of images ...")
    train_loader, val_loader, test_loader = transform_and_split_data(transform_dict, data_dir, val_ratio, test_ratio)
    visualize_training_data(train_loader, class_names)  # Optionally visualize some images


if __name__ == '__main__':
    sys.exit(load_data())
