Author: Casey Duncan
Date: 3/16/2020
Class: CSCI 508 Final
File Location: ~/csci508-final/csci508_final/Images

**OVERVIEW**
Here lies all image data that will be used for trail, training, and test puproses for this project. Currently, there are two folders within the same directory as this README file:
    1) TRIAL
       Contains a small amount of images for code testing purposes

    2) TRIANING & TEST
       Will contain al training & test data used for this project

Within each folder should be a folder called '<FOLDER NAME> IMAGES'. For example, in the TRIAL folder there should be a folder called 'TRAIL IMAGES'. Within this folder is all of the images, which should be separated into classes. For example, if there are a bunch of photos showing only glass objects, all of these images should be placed into their own folder named 'glass' which corresponds to the class 'glass'. Equivalently for photos showing only paper, metal, plastic, etc. In the future, there will be a .zip file where you can export all images within their corresponding labeled folders into this '<FOLDER NAME> IMAGES' folder.

The CNN model is trained using a pickle file which holds all of this data as an input array (X) of training & test images and an output array (Y) of labeles corresponding to each image. To create this pickle file, follow the following steps:
    1) Go to the ~/csci508-final/csci508_final directory and open the labeldata.py file in your desired text editor.

    2) Change the path to the images and path to saved files depending on if you are using the TRIAL IMAGES or the TRIANING & TEST IMAGES. 
        2a) For TRIAL IMAGES, change the following variables to the following wihtin the main function:
        path_images = cwd.joinpath('csci508_final', 'Images', 'TRIAL', 'TRIAL IMAGES')
        path_save = cwd.joinpath('csci508_final', 'Images', 'TRIAL')

        2b) For TRIANING & TEST, change the following variables to the following wihtin the main function:
        path_images = cwd.joinpath('csci508_final', 'Images', 'TRIANING & TEST', 'TRIANING & TEST IMAGES')
        path_save = cwd.joinpath('csci508_final', 'Images', 'TRIANING & TEST')

    3) Open up a new terminal window and change the terminal's working director to the following:
    ~/csci508-final

    4) Within the terminal, run the labeldata.py file by entering the following:
    poetry run python csci508_final/labeldata.py

    The following should print onto your terminal:
    "Image File # of 'Total Image Quantity'"
    For example, there are 18 images total in the TRIAL IMAGES folder, so the last line that should appear will be:
    "Image File 18 of 18"

    5) Once the labeldata.py file has finished running, there should be a LabelDict.csv file within the folder containing the folder holding all of the images and a X_Y_Data.pickle file holding all of the data. For example, if performing this on the TRAIL IMAGES the .csv and ,pickle files will be in the TRIAL folder. 
        - Opening the .csv file will show each class name and its corresponding label.
        - The .pickle file holds an input variable (X), which is an array including each training image and an output variable (Y), which is an array including the label associated with the class of each image in X.

To make sure the labels are correct, you can run the pickle_file_test.py file to cycle through each image in X and verify that its label in Y is correct. To do this, perform the following steps:
    1) Go to the ~/csci508-final/csci508_final directory and open the pickle_file_test.py file in your desired text editor.

    2) Change the path of the X_Y_Data.pickle file depending on if you are using the TRIAL IMAGES or the TRIANING & TEST IMAGES. 
        2a) For TRIAL IMAGES, change the following variables to the following wihtin the main function:
        path_save = cwd.joinpath('csci508_final', 'Images', 'TRIAL')

        2b) For TRIANING & TEST, change the following variables to the following wihtin the main function:
        path_save = cwd.joinpath('csci508_final', 'Images', 'TRIANING & TEST')

    3) Open up a new terminal window and change the terminal's working director to the following:
    ~/csci508-final

    4) Within the terminal, run the labeldata.py file by entering the following:
    poetry run python csci508_final/pickle_file_test.py

    When this has started running, each image within the X array should appear with it's corresponding label and class displayed in the title of the image. To go to the next image, press any key on the keyboard.