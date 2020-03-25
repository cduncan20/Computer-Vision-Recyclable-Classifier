import os
import numpy as np
import csv
import sys
import cv2
import pathlib
from skimage.util import random_noise
from PIL import Image
import random
import h5py


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
        item = item.lower()  # make everything lowercaseLoading Data Files
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


# Gets the names of the class types and saves them to a list
def get_class_list(path_images):
    # Define variable listing all possible classes
    file_list = os.listdir(path_images)  # All files in directory
    class_list = []  # Initialize list of all classes
    for filename in file_list:  # loop through all the files and folders
        if os.path.isdir(os.path.join(os.path.abspath(path_images),
                                      filename)):  # check whether the current object is a folder or not
            class_list.append(filename)  # Append class to list of classes

    return class_list


# Splits indices for a folder into train, validation, and test indices with random sampling
def split_data(path_images, train_ratio, valid_ratio, test_ratio):
    # Verify Training, Validation, and Test ratios are not negative
    if train_ratio < 0.0:
        sys.exit("Train Data Ratio is negative. Change to value between 0.0 and 1.0")
    if valid_ratio < 0.0:
        sys.exit("Validation Data Ratio is negative. Change to value between 0.0 and 1.0")
    if test_ratio < 0.0:
        sys.exit("Test Data Ratio is negative. Change to value between 0.0 and 1.0")

    # Verify Training, Validation, and Test ratios add to 1.0
    if train_ratio + valid_ratio + test_ratio != 1.0:
        sys.exit("Train, Validation, & Test Data Ratios do not add to equal 1.0")

    image_quantity = len(os.listdir(path_images))  # Total number of images in path_images
    all_image_indices = list(range(1, image_quantity + 1))  # List of all image indices from 1 to last image

    # Get indices of Training Images
    random.seed(1)
    train_indices = random.sample(list(range(1, image_quantity + 1)), int(train_ratio * image_quantity))

    # Get indices of remaining images for Validation & Test Images
    remaining_images = list(set(all_image_indices) - set(train_indices))
    valid_ratio_remain = image_quantity*valid_ratio/len(remaining_images)

    # Get indices of Validation Images
    random.seed(1)
    valid_indices = random.sample(remaining_images, int(valid_ratio_remain * len(remaining_images)))

    # Get indices of Test Images
    test_indices = list(set(remaining_images) - set(valid_indices))

    return train_indices, valid_indices, test_indices


# Gets file names for a particular type of class, given indices
def get_image_names(class_type, indices):
    image_file_names = [class_type + str(i) + ".jpg" for i in indices]
    return image_file_names


# Gets the names of the images for Training, Validation, and Testing
def train_valid_test_image_names(path_images, class_list, train_ratio, valid_ratio, test_ratio):
    # Initiallize arrays holding image files names for Training, Validation, and Testing
    train_names = np.empty((0, 1))
    valid_names = np.empty((0, 1))
    test_names = np.empty((0, 1))

    # Append image names to arrays
    for class_i in class_list:
        class_i_images = os.path.join(path_images, class_i)  # Path of images within class i
        train_ind, valid_ind, test_ind = split_data(class_i_images, train_ratio, valid_ratio, test_ratio)  # Indices of images for Train, Valid, & Test
        train_names = np.append(train_names, np.asarray(get_image_names(class_i, train_ind)))
        valid_names = np.append(valid_names, np.asarray(get_image_names(class_i, valid_ind)))
        test_names = np.append(test_names, np.asarray(get_image_names(class_i, test_ind)))

    return train_names, valid_names, test_names


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


# Cycle through all images and find the largest height & width
def get_max_image_size(path_images, class_list):
    height_max = 0  # Initialize maximum photo height
    width_max = 0  # Initialize maximum photo width
    for class_i in class_list:
        # Cycle through each folder within images location
        class_i_path = os.path.join(path_images, class_i)  # Define folder path of all images within class
        class_i_images = os.listdir(class_i_path)  # Create array of all images names within folder
        for class_i_image in class_i_images:
            # Cycle through each image within images location for specific class
            class_i_image_path = os.path.join(class_i_path, class_i_image)  # Define path to image 'class_i_image' within class 'class_i'
            image = Image.open(class_i_image_path)  # Get image size without reading image into memory
            [width, height] = image.size
            if height_max < height:
                height_max = height
            if width_max < width:
                width_max = width

    return width_max, height_max


# Add padding to image width & height of images smaller than largest image
def add_padding(image, width_des, height_des):
    height_img = image.shape[0]
    width_img = image.shape[1]

    left_padding = round((width_des - width_img) / 2)
    right_padding = left_padding
    if 2*left_padding + width_img < width_des:
        left_padding = left_padding + 1
    
    top_padding = round((height_des - height_img) / 2)
    bottom_padding = top_padding
    if 2*top_padding + height_img < height_des:
        top_padding = top_padding + 1

    # Using cv2.copyMakeBorder() method 
    image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT)
    return image


# Create Pickle file of input and output data
def load_dataset(path_images, path_save, dict_name, class_list, image_names, horz, vert, vert_horz, rot_45, noise, blur, is_color=True):
    # Load dictionary of labels
    dictionary = load_dict(path_save, dict_name)

    # Standard image width & height
    width, height = get_max_image_size(path_images, class_list)

    # Count number of images within training, validation, or testing
    file_count = image_names.size

    # Count number of augmentations
    aug_count = 0
    if horz:
        aug_count += 1
    if vert:
        aug_count += 1
    if vert_horz:
        aug_count += 1
    if rot_45:
        aug_count += 1
    if noise:
        aug_count += 1
    if blur:
        aug_count += 1
    
    # Define size of input and output data arrays
    if is_color:
        X = np.zeros((file_count * (aug_count+1), height, width, 3), dtype=np.uint8)  # Define standard input (X) image size
    else:
        X = np.zeros((file_count * (aug_count+1), height, width), dtype=np.uint8)  # Define standard input (X) image size
    Y = np.zeros((file_count * (aug_count+1), 1), dtype=np.uint8)  # Define standard output (Y) label size

    # Cycle through each image within each folder and create array of input and output data
    img_counter = 0  # Initialize counter to count how many images (including augmentations) have been processed
    file_counter = 0 # Initialize counter to print how many images (not including augmentations) have been processed
    for class_i_image in image_names: 
        for class_i in class_list:
            class_i = class_i.lower()  # Ensure all file names are lowercase for when finding label within .csv file

            # Define path to image 'class_i_image' within class 'class_i'
            class_i_image_path = os.path.join(path_images, class_i, class_i_image)

            # Verify path exists
            if os.path.isfile(class_i_image_path):
                image = cv2.imread(class_i_image_path)

                if not is_color:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to gray

                image_reshape = add_padding(image, width, height) # Reshape image to defined width and height
                X[img_counter] = image_reshape  # Append image to input array
                Y[img_counter] = dictionary[class_i]  # Append label of image to output array

                aug_complete_count = 0
                if horz:
                    aug_complete_count += 1
                    X[img_counter + aug_complete_count] = data_augmentation(image_reshape, horz=True)  # Append image to input array
                    Y[img_counter + aug_complete_count] = dictionary[class_i]  # Append label of image to output array

                if vert:
                    aug_complete_count += 1
                    X[img_counter + aug_complete_count] = data_augmentation(image_reshape, vert=True)  # Append image to input array
                    Y[img_counter + aug_complete_count] = dictionary[class_i]  # Append label of image to output array

                if vert_horz:
                    aug_complete_count += 1
                    X[img_counter + aug_complete_count] = data_augmentation(image_reshape, vert_horz=True)  # Append image to input array
                    Y[img_counter + aug_complete_count] = dictionary[class_i]  # Append label of image to output array

                if rot_45:
                    aug_complete_count += 1
                    X[img_counter + aug_complete_count] = data_augmentation(image_reshape, rot_45=True)  # Append image to input array
                    Y[img_counter + aug_complete_count] = dictionary[class_i]  # Append label of image to output array

                if noise:
                    image_noise = data_augmentation(image_reshape, noise=True)
                    aug_complete_count += 1

                    # image_noise is originally a float64 data type, so convert to a uint8 to match others
                    image_noise = 255 * image_noise / np.amax(image_noise)  # Current values are between 0 and 1. Change to between 0 and 255
                    image_noise = image_noise.astype(np.uint8)  # Convert to uint8 data type

                    X[img_counter + aug_complete_count] = image_noise  # Append image to input array
                    Y[img_counter + aug_complete_count] = dictionary[class_i]  # Append label of image to output array

                if blur:
                    aug_complete_count += 1
                    X[img_counter + aug_complete_count] = data_augmentation(image_reshape, blur=True)  # Append image to input array
                    Y[img_counter + aug_complete_count] = dictionary[class_i]  # Append label of image to output array

                img_counter += aug_complete_count+1  # Count total number of images saved
                file_counter += 1

                # Output to terminal number of images saved
                print("Base Image File {} of {} and transformations added to data set. X File Size = {} GB and Y File Size = {} GB \n".format(
                    file_counter, file_count, round(sys.getsizeof(X) * 1e-9, 2), round(sys.getsizeof(Y) * 1e-9, 2)))

            # If class_i_image_path does not exist, skip to next class
            else:
                pass

    return X, Y


def label_data(data_augmentation_dictionary, train_ratio, valid_ratio, test_ratio):
    # Create dictionary of labels and their corresponding labels
    cwd = pathlib.Path.cwd()
    # path_images = cwd.joinpath('csci508_final', 'Images', 'TRIAL', 'TRIAL_IMAGES')  # Path to files
    # path_save = cwd.joinpath('csci508_final', 'Images', 'TRIAL')  # Path for where to save files
    path_images = cwd.joinpath('csci508_final','Images', 'TRAINING_&_TEST', 'TRAINING_&_TEST_IMAGES') # Path to files
    path_save = cwd.joinpath('csci508_final','Images', 'TRAINING_&_TEST') # Path for where to save files
    dict_name = 'LabelDict.csv'  # File that saves classes and their corresponding labels
    create_dict(path_images, path_save, dict_name)  # Create dictionary

    # Handle data augmentation flagging logic provided by data_augmentation_dictionary
    horiz = data_augmentation_dictionary['horizontal']
    vert = data_augmentation_dictionary['vertical']
    horiz_vert = data_augmentation_dictionary['horizontal-vertical']
    rot45 = data_augmentation_dictionary['rot45']
    noise = data_augmentation_dictionary['noise']
    blur = data_augmentation_dictionary['blur']

    # Define variable listing all possible classes
    class_list = get_class_list(path_images)

    # Obtain image names for Training, Validation, & Testing
    train_names, valid_names, test_names = train_valid_test_image_names(path_images, class_list, train_ratio, valid_ratio, test_ratio)
    for i in range(1, 4):
        image_names = []
        file_name_X = []
        file_name_Y = []
        if i == 1:
            image_names = train_names
            file_name_X = 'X_Train.h5'
            file_name_Y = 'Y_Train.h5'
            print("Training Data Started")
        if i == 2:
            image_names = valid_names
            file_name_X = 'X_Valid.h5'
            file_name_Y = 'Y_Valid.h5'
            print("Validation Data Started")
        if i == 3:
            image_names = test_names
            file_name_X = 'X_Test.h5'
            file_name_Y = 'Y_Test.h5'
            print("Test Data Started")

        # Create input (X) and output(Y) data from images within designated path
        X, Y = load_dataset(path_images, path_save, dict_name, class_list, image_names, horiz, vert, horiz_vert, rot45, noise, blur)
        print("Date Loaded")

        # Save X & Y data to a .h5 file within designated path
        file_path_and_name_X = os.path.join(path_save, file_name_X)  # Define file path & name
        file_path_and_name_Y = os.path.join(path_save, file_name_Y)  # Define file path & name
        h5f = h5py.File(file_path_and_name_X, 'w')
        h5f.create_dataset(file_path_and_name_X, data=X)
        h5f = h5py.File(file_path_and_name_Y, 'w')
        h5f.create_dataset(file_path_and_name_Y, data=Y)

        print("Data Saved")


if __name__ == '__main__':
    sys.exit(label_data())