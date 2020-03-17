import cv2
import pickle
import csv
import pathlib
import os

def test_pickle(path_save):
    data_file = 'X_Y_Data.pickle'
    class_file = 'LabelDict.csv'
    data_file_path_and_name = os.path.join(path_save, data_file) # Define file path & name for pickle file
    class_file_path_and_name = os.path.join(path_save, class_file) # Define file path & name for class file
    with open(data_file_path_and_name, 'rb') as f:
        X, Y = pickle.load(f)

    class_type = []
    with open(class_file_path_and_name) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            class_type.append(row[0])

    for i in range(0, len(X)):
        image = X[i]
        label = Y[i]
        class_name = class_type[label]
        cv2.imshow("Label %d: %s" % (label, class_name), image)
        cv2.waitKey(0)


if __name__ == '__main__':
    # Create dictionary of labels and their corresponding labels
    cwd = pathlib.Path.cwd()
    path_save = cwd.joinpath('csci508_final', 'Images', 'TRIAL') # Path for where to read files

    test_pickle(path_save)