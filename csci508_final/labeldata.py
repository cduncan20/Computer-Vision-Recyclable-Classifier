# Authors: Matt Stanley, Rachel Breshears, & Casey Duncan
# Date: 12/19/2019
# The functions below go through the roughly 80,000 images of math characters within their corresponding labeled folders, and save the
# data imto the variables X (3D array of almost 80,000 images, # size 45 x 45 pixels, of 66 different math characters) and Y (2D array
# of labels corresponding to each input math character image). The data is then made into a pickle file called X_Y_Data.pickle

import os
import numpy as np
import csv
import sys
import cv2
import pickle
import pathlib

# Create dictionary of Labels
def createDict(path_images, path_save, dict_name):
    filelist = os.listdir(path_images) # All files in directory
    dirlist = []
    for filename in filelist: # loop through all the files and folders
        if os.path.isdir(os.path.join(os.path.abspath(path_images), filename)): # check whether the current object is a folder or not
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

    dict = {}
    counter = 0

    for item in multiple:
        dict[item] = counter
        counter += 1
    for item in single:
        dict[item] = counter
        counter += 1

    # writing to an Excel file
    file_path_and_name = os.path.join(path_save, dict_name) # Define file path & name
    file = open(file_path_and_name, "w") # Create file given path and name
    
    file_writer = csv.writer(file) # Initialize writer for writing values to .csv file
    for key, val in dict.items():
        file_writer.writerow([key, val]) # Write all class types and class numbers to file

    file.close() # Close file

# Load dictionary of labels
def loadDict(path_save, dict_name):
    file_path_and_name = os.path.join(path_save, dict_name) # Define file path & name
    dict = {}
    with open(file_path_and_name) as file:
        readCSV = csv.reader(file)
        for row in readCSV:
            if len(row) > 0:
                dict[row[0]] = int(row[1])
    return dict

# Create Pickle file of input and output data
def loadDataset(path_images, path_save, dict_name, rate=0.2):
    # Load dictionary of labels
    dict = loadDict(path_save, dict_name)

    # Define variable listing all possible classes
    filelist = os.listdir(path_images) # All files in directory
    class_list = [] # Initialize list of all classes
    not_class = 0 # Intialize number of files within directory not defined as a class
    for filename in filelist: # loop through all the files and folders
        if os.path.isdir(os.path.join(os.path.abspath(path_images), filename)): # check whether the current object is a folder or not
            class_list.append(filename) # Append class to list of classes
        else:
            not_class += 1

    # Count number of images within directory
    file_count = sum([len(files) for r, d, files in os.walk(path_images)]) # Count number of files in images location
    file_count = file_count - not_class # Subtract files that are not belonding to a specific class
    
    # Define size of input and output data arrays
    width = 512 # Standard image width
    height = 384 # Standard image width
    X = np.empty((0, height, width), dtype=np.uint8) # Define standard input (X) image size
    Y = np.empty((0, 1), dtype=np.uint8) # Define standard output (Y) label size

    # Cycle through each image within each folder and create array of input and output data
    counter = 0 # Initialize counter to print how many images have been processed
    for class_i in class_list: 
        # Cycle through each folder wihtin images location
        class_i_path = os.path.join(path_images, class_i) # Define folder path of all images within class
        class_i_images = os.listdir(class_i_path) # Create array of all images names within folder
        class_i = class_i.lower() # Ensure all file names are lowercase for when finding label within .csv file
        
        for class_i_image in class_i_images:
            # Cycle through each image within class_i_path and save image to X and class label to Y
            class_i_image_path = os.path.join(class_i_path, class_i_image) # Define path to image 'class_i_image' within class 'class_i'
            image = cv2.imread(class_i_image_path) # Read image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to gray
            npi = np.asarray(image).reshape(height, width)  # Reshape image to defined width and height
            X = np.append(X, [npi], axis=0)  # Append image to input array
            Y = np.append(Y, dict[class_i]) # Append label of image to output array
            counter += 1 # Count total number of images saved

            # Ouput to terminal number of images saved
            output_string = f"Image File {counter} of {file_count}\n"
            sys.stdout.write(output_string)
            sys.stdout.flush()
    
    # Split data into training and test. This can be done later too.
    # x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = rate)
    return X, Y


if __name__ == '__main__':
    # Create dictionary of labels and their corresponding labels
    cwd = pathlib.Path.cwd()
    path_images = cwd.joinpath('csci508_final', 'Images', 'TRIAL', 'TRIAL_IMAGES') # Path to files
    path_save = cwd.joinpath('csci508_final', 'Images', 'TRIAL') # Path for where to save files
    dict_name = 'LabelDict.csv' # File that saves classes and their corresponding labels
    createDict(path_images, path_save, dict_name) # Create dictionary

    # Create inpout (X) and output(Y) data from images within designated path
    X, Y = loadDataset(path_images, path_save, dict_name, rate=0.2)

    # Split data into training and test. This can be done later too.
    # x_train, x_test, y_train, y_test = loadDataset(path,dict_name,rate = 0.2)

    # Save X & Y data to a pickle file within designated path
    pickle_name = 'X_Y_Data.pickle'
    file_path_and_name = os.path.join(path_save, pickle_name) # Define file path & name
    with open(file_path_and_name, 'wb') as f:
        pickle.dump([X, Y], f)