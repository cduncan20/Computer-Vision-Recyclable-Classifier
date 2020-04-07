import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# The network will recognize digits 0..num_digits-1, inclusive.
num_digits = 5       # To recognize all digits, set to 10


def main():
    # Assuming that we are on a CUDA machine, this should print a CUDA device.
    print("Using device: ", device)

    # Load MNIST data, which are images of handwritten digits (0..9).
    # Images are 28x28, single channel. See http://yann.lecun.com/exdb/mnist/.
    # There are 60,000 training examples, and a test set of 10,000 examples.
    print("Loading MNIST database of images ...")
    train_loader, val_loader, test_loader = load_data()
    visualize_training_data(train_loader)    # Optionally visualize some images

    # Optionally look at the shape of the images and the labels.
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print(images.shape)
    print(labels.shape)

    # Create network.
    print("Creating the network ...")
    model = Net()
    model.to(device)
    print(model)

    # Count the number of trainable parameters.
    print("Trainable model parameters:")
    for p in model.parameters():
        if p.requires_grad:
            print("Tensor ", p.shape, " number of params: ", p.numel())
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters: %d" % num_params)

    # Run some random data through the network to check the output size.
    x = torch.zeros((2, 1, 28, 28)).to(device)  # minibatch size 2, image size [1, 28, 28]
    scores = model(x)
    print("Verify output size, should be (2,%d)" % num_digits)
    print(scores.size())        # Should be size (2,num_digits)

    # Train the network.
    start_time = time.time()
    learning_rate = 1e-2
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_model(model, optimizer, train_loader, val_loader, epochs=2)
    print("Total training time: %f seconds" % (time.time()-start_time))

    # Test the network.
    print('Evaluating accuracy on test set ...')
    eval_test_accuracy(test_loader, model)

    # Show some example classifications.
    print("Results on example test images:")
    show_example_results(test_loader, model)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        in_channels = 1
        h = [10, 25]    # Number of nodes, each layer
        out_channels = num_digits
        self.conv = nn.Conv2d(in_channels, h[0], kernel_size=3, padding=1)
        self.fc1 = nn.Linear(14*14*h[0], h[1])
        self.fc2 = nn.Linear(h[1], out_channels)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv(x), 2))
        x = x.view(x.shape[0], -1)      # Reshape size (batch size,C,H,W) to (batch size,C*H*W)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    # Get training and validation datasets.
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    idx_to_keep = train_dataset.targets < num_digits
    train_dataset.data = train_dataset.data[idx_to_keep]
    train_dataset.targets = train_dataset.targets[idx_to_keep]

    num_images = len(train_dataset)
    num_train = int(0.8 * num_images)
    num_val = num_images - num_train
    train_set, val_set = torch.utils.data.random_split(train_dataset, [num_train, num_val])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True, num_workers=4)

    # Get test dataset.
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    idx_to_keep = test_dataset.targets < num_digits
    test_dataset.data = test_dataset.data[idx_to_keep]
    test_dataset.targets = test_dataset.targets[idx_to_keep]
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

    return train_loader, val_loader, test_loader


def train_model(model, optimizer, train_loader, val_loader, epochs=1):
    model = model.to(device=device)  # move the model parameters to CPU or GPU
    for e in range(epochs):
        print("Epoch: ", e)
        for t, (x, y) in enumerate(train_loader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % 100 == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_val_accuracy(val_loader, model)
                print()


def visualize_training_data(train_loader):
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # print labels
    L = labels.numpy()
    L = np.reshape(L, (8,8))    # Assume batch is size 64
    for r in range(8):
        out_string = ""
        for c in range(8):
            out_string += "%d " % L[r,c]
        print(out_string)

    # show images
    imshow(torchvision.utils.make_grid(images))

# Function to show an image.
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def check_val_accuracy(val_loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%f) on validation' % (num_correct, num_samples, acc))


def eval_test_accuracy(test_loader, model):
    # Create the confusion matrix.
    confusion_matrix = np.zeros((num_digits, num_digits))
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            for t, p in zip(y.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix.astype(int))

    print("Accuracy per class:")
    # This is the number of times a digit was detected correctly, divided by
    # the total number of times that digit was presented to the system.
    print(confusion_matrix.diagonal()/confusion_matrix.sum(1))

    print("Overall accuracy:")
    print(np.sum(confusion_matrix.diagonal())/np.sum(confusion_matrix))


def show_example_results(loader, model):
    dataiter = iter(loader)
    images, labels = dataiter.next()

    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        x = images
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        scores = model(x)
        max_scores, preds = scores.max(1)

        # print true labels
        print("True labels:")
        L = labels.numpy()
        L = np.reshape(L, (8, 8))  # Assume batch is size 64
        for r in range(8):
            out_string = ""
            for c in range(8):
                out_string += "%d " % L[r, c]
            print(out_string)

        # print predicted labels
        print("Predicted labels:")
        L = preds.view(8, 8)  # Reshape to 8x8 array
        for r in range(8):
            out_string = ""
            for c in range(8):
                out_string += "%d " % L[r, c]
            print(out_string)

        # print scores
        print("Scores:")
        S = max_scores.view(8, 8)  # Reshape to 8x8 array
        for r in range(8):
            out_string = ""
            for c in range(8):
                out_string += "%.3f " % S[r, c].item()
            print(out_string)

    imshow(torchvision.utils.make_grid(images))

if __name__ == "__main__":
    main()