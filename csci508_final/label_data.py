import os
import numpy as np
import csv
import sys
import cv2
import pickle
import pathlib
from skimage.util import random_noise


# Create dictionary of Labels
def create_dict(path_images, path_save, dict_name):
    filelist = os.listdir(path_images)  # All files in directory
    dirlist = []
    for filename in filelist:  # loop through all the files and folders
        # check whether the current object is a folder or not
        if os.path.isdir(os.path.join(os.path.abspath(path_images), filename)):
            dirlist.append(filename)

    single = []
    multiple = []

    for item in dirlist:
        item = item.lower()  # make everything lowercase
        if len(item) == 1:
            single.append(item)
        else:
            multiple.append(item)

    multiple.sort()  # alphabetical order
    single.sort()  # ascii numerical order

    dictionary = {}
    counter = 0

    for item in multiple:
        dictionary[item] = counter
        counter += 1
    for item in single:
        dictionary[item] = counter
        counter += 1

    # writing to an Excel file
    file_path_and_name = os.path.join(path_save, dict_name)  # Define file path & name
    file = open(file_path_and_name, "w")  # Create file given path and name
    
    file_writer = csv.writer(file)  # Initialize writer for writing values to .csv file
    for key, val in dictionary.items():
        file_writer.writerow([key, val])  # Write all class types and class numbers to file

    file.close()  # Close file


# Load dictionary of labels
def load_dict(path_save, dict_name):
    file_path_and_name = os.path.join(path_save, dict_name)  # Define file path & name
    dictionary = {}
    with open(file_path_and_name) as file:
        readCSV = csv.reader(file)
        for row in readCSV:
            if len(row) > 0:
                dictionary[row[0]] = int(row[1])
    return dictionary


# Create augmented photos
def data_augmentation(image, horz=False, vert=False, vert_horz=False, rot_45=False, noise=False, blur=False):
    if horz:
        # Create horizontally flipped image
        img_flip_horz = cv2.flip(image, 1)
        return img_flip_horz

    if vert:
        # Create vertically flipped image
        img_flip_vert = cv2.flip(image, 0)
        return img_flip_vert

    if vert_horz:
        # Create vertically & horizontally flipped image
        img_flip_vert_horz = cv2.flip(image, -1)
        return img_flip_vert_horz

    if rot_45:
        # Create rotate 45 degrees image
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, 45, 1.0)
        img_rot_45 = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return img_rot_45

    if noise:
        # Create image with random noise
        image_noise = random_noise(image)
        return image_noise

    if blur:
        # Create image with gaussian blur
        image_blur = cv2.GaussianBlur(image, (11,11),0)
        return image_blur


# Create Pickle file of input and output data
def load_dataset(path_images, path_save, dict_name, is_color=True, horz=True, vert=True, vert_horz=True, rot_45=True, noise=True, blur=True):
    # Load dictionary of labels
    dictionary = load_dict(path_save, dict_name)

    # Define variable listing all possible classes
    filelist = os.listdir(path_images)  # All files in directory
    class_list = []  # Initialize list of all classes
    not_class = 0  # Initialize number of files within directory not defined as a class
    for filename in filelist:  # loop through all the files and folders
        # check whether the current object is a folder or not
        if os.path.isdir(os.path.join(os.path.abspath(path_images), filename)):
            class_list.append(filename)  # Append class to list of classes
        else:
            not_class += 1

    # Count number of images within directory
    file_count = sum([len(files) for r, d, files in os.walk(path_images)])  # Count number of files in images location
    file_count = file_count - not_class  # Subtract files that are not belonging to a specific class
    
    # Define size of input and output data arrays
    width = 512  # Standard image width
    height = 384  # Standard image width
    if is_color:
        X = np.empty((0, height, width, 3), dtype=np.uint8)  # Define standard input (X) image size
    else:
        X = np.empty((0, height, width), dtype=np.uint8)  # Define standard input (X) image size
    Y = np.empty((0, 1), dtype=np.uint8)  # Define standard output (Y) label size

    # Cycle through each image within each folder and create array of input and output data
    counter = 0  # Initialize counter to print how many images have been processed
    for class_i in class_list: 
        # Cycle through each folder within images location
        class_i_path = os.path.join(path_images, class_i)  # Define folder path of all images within class
        class_i_images = os.listdir(class_i_path)  # Create array of all images names within folder
        class_i = class_i.lower()  # Ensure all file names are lowercase for when finding label within .csv file

        # Cycle through each image within class_i_path and save image to X and class label to Y
        for class_i_image in class_i_images:
            # Define path to image 'class_i_image' within class 'class_i'
            class_i_image_path = os.path.join(class_i_path, class_i_image)
            image = cv2.imread(class_i_image_path)

            if is_color:
                image_reshape = np.asarray(image).reshape(height, width, 3)
            else:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to gray
                image_reshape = np.asarray(image_gray).reshape(height, width)

            X = np.append(X, [image_reshape], axis=0)  # Append image to input array
            Y = np.append(Y, dictionary[class_i])  # Append label of image to output array

            if horz:
                img_flip_horz = data_augmentation(image_reshape, horz=True)
                X = np.append(X, [img_flip_horz], axis=0)  # Append image to input array
                Y = np.append(Y, dictionary[class_i])  # Append label of image to output array

            if vert:
                img_flip_vert = data_augmentation(image_reshape, vert=True)
                X = np.append(X, [img_flip_vert], axis=0)  # Append image to input array
                Y = np.append(Y, dictionary[class_i])  # Append label of image to output array

            if vert_horz:
                img_flip_vert_horz = data_augmentation(image_reshape, vert_horz=True)
                X = np.append(X, [img_flip_vert_horz], axis=0)  # Append image to input array
                Y = np.append(Y, dictionary[class_i])  # Append label of image to output array

            if rot_45:
                img_rot_45 = data_augmentation(image_reshape, rot_45=True)
                X = np.append(X, [img_rot_45], axis=0)  # Append image to input array
                Y = np.append(Y, dictionary[class_i])  # Append label of image to output array

            if noise:
                image_noise = data_augmentation(image_reshape, noise=True)

                # image_noise is originally a float64 data type, so convert to a uint8 to match others
                # Current values are between 0 and 1. Change to between 0 and 255
                image_noise = 255 * image_noise / np.amax(image_noise)
                image_noise = image_noise.astype(np.uint8)  # Convert to uint8 data type

                X = np.append(X, [image_noise], axis=0)  # Append image to input array
                Y = np.append(Y, dictionary[class_i])  # Append label of image to output array

            if blur:
                image_blur = data_augmentation(image_reshape, blur=True)
                X = np.append(X, [image_blur], axis=0)  # Append image to input array
                Y = np.append(Y, dictionary[class_i])  # Append label of image to output array

            counter += 1  # Count total number of images saved

            # Output to terminal number of images saved
            output_string = "Image File {} of {}\n".format(counter, file_count)
            sys.stdout.write(output_string)
            sys.stdout.flush()

    return X, Y


def label_data():
    # Create dictionary of labels and their corresponding labels
    cwd = pathlib.Path.cwd()
    path_images = cwd.joinpath('csci508_final', 'Images', 'TRIAL', 'TRIAL_IMAGES')  # Path to files
    path_save = cwd.joinpath('csci508_final', 'Images', 'TRIAL')  # Path for where to save files
    dict_name = 'LabelDict.csv'  # File that saves classes and their corresponding labels
    create_dict(path_images, path_save, dict_name)  # Create dictionary

    # Create input (X) and output(Y) data from images within designated path
    X, Y = load_dataset(path_images, path_save, dict_name)

    # Save X & Y data to a pickle file within designated path
    pickle_name = 'X_Y_Data.pickle'
    file_path_and_name = os.path.join(path_save, pickle_name) # Define file path & name
    with open(file_path_and_name, 'wb') as f:
        pickle.dump([X, Y], f)


if __name__ == '__main__':
    sys.exit(label_data())