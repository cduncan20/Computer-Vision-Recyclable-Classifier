Author: Casey Duncan
Date: 3/17/2020
Class: CSCI 508 Final
File Location: ~/csci508-final/csci508_final/Images

**OVERVIEW**
Here lies all image data that will be used for trial, training, and test purposes for this project. Currently, there are two folders within the same directory as this README file:
   	1) TRIAL
	Contains a small amount of images for code testing purposes

    	2) TRAINING_&_TEST
	Will contain all training & test data used for this project

Within each folder should be a folder called '<FOLDER NAME>_IMAGES'. For example, in the TRIAL folder there should be a folder called 'TRIAL_IMAGES'. Within this folder is all of the images, which should be separated into classes. For example, if there are a bunch of photos showing only glass objects, all of these images should be placed into their own folder named 'glass' which corresponds to the class 'glass'. Equivalently for photos showing only paper, metal, plastic, etc. In the future, there will be a .zip file where you can export all images within their corresponding labeled folders into this '<FOLDER NAME>_IMAGES' folder.

The CNN model is trained using a pickle file which holds all of this data as an input array (X) of training & test images and an output array (Y) of labeles corresponding to each image. To create this pickle file, follow the following steps:
	1) Go to the ~/csci508-final/csci508_final directory and open the labeldata.py file in your desired text editor.
	
	2) Change the path to the images and path to saved files depending on if you are using the TRIAL_IMAGES or the TRAINING_&_TEST_IMAGES. 
		2a) For TRIAL_IMAGES, change the following variables to the following wihtin the main function (lines 130 & 131):
		path_images = cwd.joinpath('csci508_final', 'Images', 'TRIAL', 'TRIAL_IMAGES')
		path_save = cwd.joinpath('csci508_final', 'Images', 'TRIAL')

		2b) For TRAINING_&_TEST, change the following variables to the following wihtin the main function (lines 130 & 131):
		path_images = cwd.joinpath('csci508_final', 'Images', 'TRAINING_&_TEST', 'TRAINING_&_TEST_IMAGES')
		path_save = cwd.joinpath('csci508_final', 'Images', 'TRAINING_&_TEST')

	3) Choose whether you want the photos input into X to be color or black & white. Change the variable 'is_color' (line 137) to be equal to 'True' for saving color images to X or 'False for saving black and white images to X. The default value is 'True'.

	4) Choose which data augmentation methods you want applied to each image.
		4a) For implementing a horizontal flip augmentation, change the variable 'horz' to 'False'. The default is 'True'.
		4b) For implementing a vertical flip augmentation, change the variable 'vert' to 'False'. The default is 'True'.
		4c) For implementing a vertical & horizontal flip augmentation, change the variable 'vert_horz' to 'False'. The default is 'True'.
		4d) For implementing a 45 degree rotation augmentation, change the variable 'rot_45' to 'False'. The default is 'True'.
		4e) For implementing a random noise augmentation, change the variable 'blur' to 'False'. The default is 'True'.
		4f) For implementing a gaussian blur augmentation, change the variable 'gaussian' to 'False'. The default is 'True'.

	5) Open up a new terminal window and change the terminal's working director to the following:
~/csci508-final

	6) Within the terminal, run the labeldata.py file by entering the following:
poetry run python csci508_final/labeldata.py

	The following should print onto your terminal:
	"Image File # of <Total Image Quantity>"
	For example, there are 18 images total in the TRIAL_IMAGES folder, so the last line that should appear will be:
	"Image File 18 of 18"

	7) Once the labeldata.py file has finished running, there should be a LabelDict.csv file within the folder containing the folder holding all of the images and a X_Y_Data.pickle file holding all of the data. For example, if performing this on the TRIAL_IMAGES the .csv and ,pickle files will be in the TRIAL folder. 
        - Opening the .csv file will show each class name and its corresponding label.
        - The .pickle file holds an input variable (X), which is an array including each training image and an output variable (Y), which is an array including the label associated with the class of each image in X.

To make sure the labels are correct, you can run the pickle_file_test.py file to cycle through each image in X and verify that its label in Y is correct. To do this, perform the following steps:
	1) Go to the ~/csci508-final/csci508_final directory and open the pickle_file_test.py file in your desired text editor.

	2) Change the path of the X_Y_Data.pickle file depending on if you are using the TRIAL_IMAGES or the TRAINING_&_TEST_IMAGES. 
        	2a) For TRIAL_IMAGES, change the following variables to the following wihtin the main function:
        	path_save = cwd.joinpath('csci508_final', 'Images', 'TRIAL')
		
        	2b) For TRAINING_&_TEST, change the following variables to the following wihtin the main function:
        	path_save = cwd.joinpath('csci508_final', 'Images', 'TRAINING_&_TEST')

    	3) Open up a new terminal window and change the terminal's working director to the following:
    	~/csci508-final

    	4) Within the terminal, run the labeldata.py file by entering the following:
    	poetry run python csci508_final/pickle_file_test.py

    	When this has started running, each image within the X array should appear with it's corresponding label and class displayed in the title of the image. To go to the next image, press any key on the keyboard.
