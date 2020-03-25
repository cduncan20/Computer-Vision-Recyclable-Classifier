import cv2
import csv
import pathlib
import sys
import os
import h5py

def test_data_file(path_save):
    X_file = 'X_Train.h5'  # X Data file name
    Y_file = 'Y_Train.h5'  # Y Data file name
    class_file = 'LabelDict.csv'
    data_file_path_and_name_X = os.path.join(path_save, X_file)    # Define file path & name for .h5 file
    data_file_path_and_name_Y = os.path.join(path_save, Y_file)  # Define file path & name for .5 file
    class_file_path_and_name = os.path.join(path_save, class_file)  # Define file path & name for class file

    # Read Files
    print("Loading Data Files")
    h5f = h5py.File(data_file_path_and_name_X, 'r')
    X = h5f[data_file_path_and_name_X][:]
    h5f.close()
    h5f = h5py.File(data_file_path_and_name_Y, 'r')
    Y = h5f[data_file_path_and_name_Y][:]
    h5f.close()

    print("Images Ready. Press Any key to see next image")

    class_type = []
    with open(class_file_path_and_name) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            class_type.append(row[0])

    for i in range(0, len(X)):
        image = X[i]
        label = Y[i][0]
        class_name = class_type[label]
        cv2.imshow("Label %d: %s" % (label, class_name), image)
        cv2.waitKey(0)


def run_test():
    # Create dictionary of labels and their corresponding labels
    cwd = pathlib.Path.cwd()
    # path_save = cwd.joinpath('csci508_final', 'Images', 'TRIAL')  # Path for where to read files
    path_save = cwd.joinpath('csci508_final','Images', 'TRAINING_&_TEST') # Path for where to save files

    test_data_file(path_save)


if __name__ == '__main__':
    sys.exit(run_test())
