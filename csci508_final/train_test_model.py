import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
import pathlib
import csv
import os
import sys

from .load_data import load_data
from .architectures.model_loader import model_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cwd = pathlib.Path.cwd()
model_save_path = cwd.joinpath("csci508_final", "saved_models_and_results", "saved_models")
model_results_save_path = cwd.joinpath("csci508_final", "saved_models_and_results", "saved_results")


def main(transform_dict, model_dict, model_file_name, train_ratio, val_ratio, test_ratio, epochs):
    # Assuming that we are on a CUDA machine, this should print a CUDA device.
    print("Using device: ", device)
    print("")

    # Load Data
    print("Loading database of images ...")
    train_loader, val_loader, test_loader, class_names = load_data(transform_dict, val_ratio, test_ratio)
    visualize_training_data(train_loader, class_names)  # Optionally visualize some images
    num_classes = len(class_names)
    print("")

    # Create network.
    # model = Net(num_classes)
    model = model_loader(model_dict, num_classes)
    model.to(device)
    # print(model)

    # Count the number of trainable parameters.
    print("Trainable model parameters:")
    for p in model.parameters():
        if p.requires_grad:
            print("Tensor ", p.shape, " number of params: ", p.numel())
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters: %d" % num_params)
    print("")

    # Run some random data through the network to check the output size.
    x = torch.zeros((2, 3, 256, 341)).to(device)  # minibatch size 2, image size [3, 256, 341]
    scores = model(x)
    print("Quick test to verify output size: should be [2, %d]:" % num_classes)
    print(scores.size())        # Should be size (2,num_classes)
    print("")

    # Train the network.
    print("Training the network ...")
    start_time = time.time()
    train_model(model, model_file_name, train_loader, val_loader, epochs=epochs)
    print("Total training time: %f seconds" % (time.time()-start_time))
    print("")

    # Test the network.
    print('Evaluating accuracy on test set ...')
    confusion_matrix = eval_test_accuracy(test_loader, model, class_names)
    write_to_file(confusion_matrix, model_file_name, class_names, transform_dict, train_ratio, val_ratio, test_ratio, epochs)
    print("")

    # Show some example classifications.
    print("Results on example test images:")
    show_example_results(test_loader, model, class_names)
    print("")

    print("All done!")


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
    print("Classes of Images Shown:")
    out_string = out_string.split
    for image in range(len(out_string())):
        print("{}) {}".format(image+1, out_string()[image]))

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


def train_model(model, model_file_name, train_loader, val_loader, epochs):
    learning_rate = 1e-3
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print("Epoch: {0} / {1}".format(e+1, epochs))
        for t, (x, y) in enumerate(train_loader):
            model.train()  # put model to training mode
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
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
    
    PATH = cwd.joinpath(model_save_path, model_file_name + ".pth")
    torch.save(model, PATH)
    print("Model saved! Model path is shown below:")
    print(PATH)


def check_val_accuracy(val_loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%f) on validation' % (num_correct, num_samples, acc))


def eval_test_accuracy(test_loader, model, class_names):
    # Create the confusion matrix.
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes))
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)
            scores = model(x)
            _, preds = scores.max(1)
            for t, p in zip(y.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix.astype(int))

    print("Accuracy per class:")
    # This is the number of times a class was detected correctly, divided by
    # the total number of times that class was presented to the system.
    print(confusion_matrix.diagonal()/confusion_matrix.sum(1))

    print("Overall accuracy:")
    print(np.sum(confusion_matrix.diagonal())/np.sum(confusion_matrix))

    return confusion_matrix


def write_to_file(confusion_matrix, model_file_name, class_names, transform_dict, train_ratio, val_ratio, test_ratio, epochs):
    # name of csv file
    file_name = cwd.joinpath(model_results_save_path, model_file_name + "_results.csv")

    class_accuracy = [confusion_matrix.diagonal() / confusion_matrix.sum(1)]
    overall_accuracy = np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix)

    # writing to csv file
    with open(file_name, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # Write Model file name
        csvwriter.writerow(["Model file name:", model_file_name])
        csvwriter.writerows(' ')

        # Write Selected Tranforms
        csvwriter.writerow(["Selected Transforms:"])
        count = 1
        if transform_dict['horizontal']:
            csvwriter.writerow([str(count) +") Random Horizontal Flip"])
            count += 1
        if transform_dict['vertical']:
            csvwriter.writerow([str(count) +") Random Vertical Flip"])
            count += 1
        if transform_dict['rot30']:
            csvwriter.writerow([str(count) +") Random +/-30 Degree Rotation"])
            count += 1
        if transform_dict['noise']:
            csvwriter.writerow([str(count) +") Noise"])
            count += 1
        if transform_dict['blur']:
            csvwriter.writerow([str(count) +") Blur"])
            count += 1
        csvwriter.writerows(' ')

        # Write Data Split Info
        csvwriter.writerow(["Training Ratio:", train_ratio])
        csvwriter.writerow(["Validation Ratio:", val_ratio])
        csvwriter.writerow(["Testing Ratio:", test_ratio])
        csvwriter.writerows(' ')

        # Write Epoch Info
        csvwriter.writerow(["Epoch Quantity:", epochs])
        csvwriter.writerows(' ')
    
        # Write Confusion Matrix
        csvwriter.writerow(["Confusion Matrix"])
        csvwriter.writerow(class_names)
        csvwriter.writerows(confusion_matrix.astype(int))
        csvwriter.writerows(' ')

        # Write Class Accuracies
        csvwriter.writerow(["Class Accuracy"])
        csvwriter.writerow(class_names)
        csvwriter.writerows(class_accuracy)
        csvwriter.writerows(' ')

        # Write Overall Accuracy
        csvwriter.writerow(["Overall Accuracy:", overall_accuracy])
    
    print("Model results saved! Model results path is shown below:")
    print(file_name)


def show_example_results(loader, model, class_names):
    dataiter = iter(loader)
    images, labels = dataiter.next()

    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        x = images
        x = x.to(device=device)  # move to device, e.g. GPU
        scores = model(x)
        max_scores, preds = scores.max(1)

        # print true labels
        print("True labels:")
        L = labels.numpy()
        out_string = ""
        for i in range(len(L)):
            out_string += "%s " % class_names[L[i]]
        print(out_string)

        # print predicted labels
        print("Predicted labels:")
        out_string = ""
        for i in range(len(preds)):
            out_string += "%s " % class_names[preds[i].item()]
        print(out_string)

        # print scores
        print("Scores:")
        out_string = ""
        for i in range(len(max_scores)):
            out_string += "%.2f " % max_scores[i].item()
        print(out_string)

    imshow(torchvision.utils.make_grid(images))


if __name__ == '__main__':
    sys.exit(main())
