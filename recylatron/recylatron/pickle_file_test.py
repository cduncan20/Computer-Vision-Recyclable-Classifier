import cv2
import pickle
import csv

data_file = 'X_Y_Data.pickle'
class_file = 'LabelDict.csv'
with open(data_file, 'rb') as f:
    X, Y = pickle.load(f)

class_type = []
with open(class_file) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        class_type.append(row[0])

for i in range(0, len(X)):
    image = X[i]
    label = Y[i]
    class_name = class_type[label]
    cv2.imshow("Label %d: %s" % (label, class_name), image)
    cv2.waitKey(0)