# CSC-391_Project_4
Object Detection Part 2

slicer.py the program I used to generate test images, it will divide up an image (specified with target_name in the code) into 200x200 patches, and then prompt the user on whether the image features a palm tree or not, press 'A' for Palm, 'S' for skip, 'D' Not Palm. The program will automatically name each image and put it in the correct folder for later use.

LBP_Classifier.py is the program I used to run the Kullback-Leibler Divergence matching using the test set of images. Images must be in the Sample_Date/Palm and Sample_Data/NotPalm folders. It splits up the data into training and test sets, with sizes depending on the settings in the file. It then runs the matching program on the images.

LBP_Display.py uses the same image files, except it uses the image data on a larger image and classifies each one and tints them either red or green for negative or positive matches to a palm image, then saves the result as a .jpg. Potential images for marking must be in the Large_Images folder.
