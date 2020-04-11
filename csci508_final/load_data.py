import torch
from torchvision import datasets, transforms
import os


def load_data():
    # Note on RandomResizedCrop: a crop of random size (default: of 0.08 to 1.0) of the
    # original size and a random aspect ratio (default: of 3/4 to 4/3) of the original
    # aspect ratio is made. This crop is finally resized to given size.
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(root=os.path.join(data_dir), transform=train_transform)
    num_images = len(dataset)
    num_train = int(0.8 * num_images)
    num_val = num_images - num_train
    train_set, val_set = torch.utils.data.random_split(dataset, [num_train, num_val])
    print("Number of training images: %d" % num_train)
    print("Number of validation images: %d" % num_val)

    test_set = datasets.ImageFolder(root=os.path.join(data_dir), transform=test_transform)
    print("Number of test images: %d" % len(test_set))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=True, num_workers=4)

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
    print(out_string)

    # show images
    imshow(torchvision.utils.make_grid(images))


def get_class_names(data_dir):



data_dir = "Images/TRAINING_&_TEST/TRAINING_&_TEST_IMAGES"
class_names = get_class_names(data_dir)

# Load data
print("Loading database of images ...")
train_loader, val_loader, test_loader = load_data(data_dir)
visualize_training_data(train_loader, class_names)    # Optionally visualize some images