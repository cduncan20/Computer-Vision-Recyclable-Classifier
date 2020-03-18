import os
import numpy as np
import csv
import sys
import cv2
import pickle
import pathlib


# Create dictionary of Labels
def create_dict(path_images, path_save, dict_name):
    filelist = os.listdir(path_images)  # All files in directory
    dirlist = []
    for filename in filelist:  # loop through all the files and folders
        if os.path.isdir(os.path.join(os.path.abspath(path_images), filename)):  # check whether the current object is a folder or not
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
    file_path_and_name = os.path.join(path_save, dict_name) # Define file path & name
    file = open(file_path_and_name, "w")  # Create file given path and name
    
    file_writer = csv.writer(file) # Initialize writer for writing values to .csv file
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


# Create Pickle file of input and output data
def load_dataset(path_images, path_save, dict_name, is_color=True):
    # Load dictionary of labels
    dictionary = load_dict(path_save, dict_name)

    # Define variable listing all possible classes
    filelist = os.listdir(path_images) # All files in directory
    class_list = [] # Initialize list of all classes
    not_class = 0 # Initialize number of files within directory not defined as a class
    for filename in filelist: # loop through all the files and folders
        if os.path.isdir(os.path.join(os.path.abspath(path_images), filename)): # check whether the current object is a folder or not
            class_list.append(filename) # Append class to list of classes
        else:
            not_class += 1

    # Count number of images within directory
    file_count = sum([len(files) for r, d, files in os.walk(path_images)]) # Count number of files in images location
    file_count = file_count - not_class # Subtract files that are not belonging to a specific class
    
    # Define size of input and output data arrays
    width = 512 # Standard image width
    height = 384 # Standard image width
    if is_color:
        X = np.empty((0, height, width, 3), dtype=np.uint8) # Define standard input (X) image size
    else:
        X = np.empty((0, height, width), dtype=np.uint8) # Define standard input (X) image size
    Y = np.empty((0, 1), dtype=np.uint8) # Define standard output (Y) label size

    # Cycle through each image within each folder and create array of input and output data
    counter = 0  # Initialize counter to print how many images have been processed
    for class_i in class_list: 
        # Cycle through each folder within images location
        class_i_path = os.path.join(path_images, class_i)  # Define folder path of all images within class
        class_i_images = os.listdir(class_i_path)  # Create array of all images names within folder
        class_i = class_i.lower()  # Ensure all file names are lowercase for when finding label within .csv file
        
        for class_i_image in class_i_images:
            # Cycle through each image within class_i_path and save image to X and class label to Y
            class_i_image_path = os.path.join(class_i_path, class_i_image) # Define path to image 'class_i_image' within class 'class_i'
            image = cv2.imread(class_i_image_path) # Read image

            if is_color:
                npi = np.asarray(image).reshape(height, width, 3)  # Reshape image to defined width and height
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to gray
                npi = np.asarray(image).reshape(height, width)  # Reshape image to defined width and height

            X = np.append(X, [npi], axis=0)  # Append image to input array
            Y = np.append(Y, dictionary[class_i])  # Append label of image to output array
            counter += 1 # Count total number of images saved

            # Ouput to terminal number of images saved
            output_string = "Image File {} of {}\n".format(counter, file_count)
            sys.stdout.write(output_string)
            sys.stdout.flush()

    return X, Y


def label_data():
    # Create dictionary of labels and their corresponding labels
    cwd = pathlib.Path.cwd()
    path_images = cwd.joinpath('csci508_final', 'Images', 'TRIAL', 'TRIAL_IMAGES') # Path to files
    path_save = cwd.joinpath('csci508_final', 'Images', 'TRIAL') # Path for where to save files
    dict_name = 'LabelDict.csv' # File that saves classes and their corresponding labels
    create_dict(path_images, path_save, dict_name) # Create dictionary

    # Create inpout (X) and output(Y) data from images within designated path
    X, Y = load_dataset(path_images, path_save, dict_name)

    # Split data into training and test. This can be done later too.
    # x_train, x_test, y_train, y_test = loadDataset(path,dict_name,rate = 0.2)
    # Save X & Y data to a pickle file within designated path
    pickle_name = 'X_Y_Data.pickle'
    file_path_and_name = os.path.join(path_save, pickle_name) # Define file path & name
    with open(file_path_and_name, 'wb') as f:
        pickle.dump([X, Y], f)


if __name__ == '__main__':
    sys.exit(label_data())