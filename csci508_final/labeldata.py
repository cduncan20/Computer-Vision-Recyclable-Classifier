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
    dirlist = os.listdir(path_images)

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
    
    w = csv.writer(file) # Initialize writer
    for key, val in dict.items():
        w.writerow([key, val]) # Write all class types and class numbers to file

    file.close()

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
    dict = loadDict(path_save, dict_name) # Load dictionary of labels
    ds1 = os.listdir(path_images) # Define path of images 
    file_count = sum([len(files) for r, d, files in os.walk(path_images)]) # Count number of files in images location
    
    # Cycle through each image within each folder and create array of input and output data
    width = 384
    height = 512
    X = np.empty((0, width, height), dtype=np.uint8) # Define standard input (X) image size
    Y = np.empty((0, 1), dtype=np.uint8) # Define standard output (Y) label size
    counter = 0 # Initialize counter to print how many images have been processed
    for d in ds1: # Cycle through each folder wihtin images location
        folder = os.path.join(path_images, d) # Define folder path of all images within class
        ds2 = os.listdir(folder) # Create array of all images names within folder
        d = d.lower() # Ensure all file names are lowercase for when finding label within .csv file
        for d2 in ds2:
            filei = os.path.join(folder, d2) # Define path to image 'd2' within class 'ds2'
            image = cv2.imread(filei) # Read image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to gray
            npi = np.asarray(image).reshape(width, height)  # Reshape image to defined width and height
            X = np.append(X, [npi], axis=0)  # Append image to input array
            Y = np.append(Y, dict[d]) # Append label of image to output array
            counter += 1
            output_string = f"Image File {counter} of {file_count}\n"
            sys.stdout.write(output_string)
            sys.stdout.flush()
    # x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = rate)
    return X, Y


if __name__ == '__main__':
    # Create dictionary of labels and their corresponding labels
    cwd = pathlib.Path.cwd()
    path_images = cwd.joinpath('csci508_final', 'Images', 'TRIAL', 'TRIAL IMAGES') # Path to files
    path_save = cwd.joinpath('csci508_final', 'Images', 'TRIAL') # Path for where to save files
    dict_name = 'LabelDict.csv' # File that saves classes and their corresponding labels
    createDict(path_images, path_save, dict_name) # Create dictionary

    # Create inpout (X) and output(Y) data from images within designated path
    X, Y = loadDataset(path_images, path_save, dict_name, rate=0.2)
    # x_train, x_test, y_train, y_test = loadDataset(path,dict_name,rate = 0.2)

    # Save X & Y data to a pickle file within designated path
    pickle_name = 'X_Y_Data.pickle'
    file_path_and_name = os.path.join(path_save, pickle_name) # Define file path & name
    with open(file_path_and_name, 'wb') as f:
        pickle.dump([X, Y], f)