import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split

from skimage.feature import local_binary_pattern

#Name: Zander Miller
#CSC-391: Project 4



#Settings
TARGET_IMAGE = "DSC08606.JPG" #Located in the Large_Images folder adjacent to this program
PALM_TEST_SIZE = 0.25 #proportion of the palm data to be used in the test set (not training)
NOTPALM_TEST_SIZE = 0.5 #proportion of the NotPalm data to be used in the test set (not training)
PATCH_SIZE = 200
#LBP Settings
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

display_dict = {True: "Is a Palm.", False: "Is not a Palm." }

#Getting the images
print("Identifying Palm Images...")
Palm_list = glob("Sample_Data/Palm/*")
print("Identifying NotPalm Images...")
NotPalm_list = glob("Sample_Data/NotPalm/*")

Palm_refs_train = []
NotPalm_refs_train = []

print("Dividing Dataset...")
ptr_list, pte_list = train_test_split(Palm_list, test_size=PALM_TEST_SIZE)
nptr_list, npte_list = train_test_split(NotPalm_list, test_size=NOTPALM_TEST_SIZE)
print("Training Palm References...")
for file in ptr_list:
    infile = cv2.imread(file, 0)
    Palm_refs_train.append(local_binary_pattern(infile, n_points, radius, METHOD))
print("Training NotPalm References...")
for file in nptr_list:
    infile = cv2.imread(file, 0)
    NotPalm_refs_train.append(local_binary_pattern(infile, n_points, radius, METHOD))

image = cv2.imread("Large_Images/"+TARGET_IMAGE)
length = image.shape[0]
width = image.shape[1]
l_max = int(length/PATCH_SIZE)
w_max = int(width/PATCH_SIZE)
print(f"Testing {w_max * l_max} patches...")

def kullback_leibler_divergence(p, q):
    #The lower the value the better the match for this
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

def matching(Palm_refs, NotPalm_refs, lbp):
    matched_value = 10
    isPalm = False
    n_bins = int(lbp.max() + 1)
    #get histogram of test image
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    #checking all of the palm images
    for ref in Palm_refs:
        #get histogram for reference image
        ref_hist, _ = np.histogram(ref, density=True, bins=n_bins, range=(0, n_bins))
        value = kullback_leibler_divergence(hist, ref_hist)
        #checking for the best match image texture
        if value < matched_value:
            matched_value = value
            isPalm = True
    for ref in NotPalm_refs:
        #get histogram for reference image
        ref_hist, _ = np.histogram(ref, density=True, bins=n_bins, range=(0, n_bins))
        value = kullback_leibler_divergence(hist, ref_hist)
        #checking for the best match image texture
        if value < matched_value:
            matched_value = value
            isPalm = False
    return isPalm

out_image = np.zeros(image.shape)

for i in range (0, l_max - 1):
    for j in range (0, w_max - 1):
        sub_set = image[i*PATCH_SIZE:(i+1)*PATCH_SIZE, j*PATCH_SIZE:(j+1)*PATCH_SIZE, :]
        g_sub_set = cv2.cvtColor(sub_set, cv2.COLOR_BGR2GRAY)
        sub_ref = local_binary_pattern(g_sub_set, n_points, radius, METHOD)
        result = matching(Palm_refs_train, NotPalm_refs_train, sub_ref)
        sub_set[:, :, 0] = sub_set[:, :, 0]/2
        print(f"({i}, {j}) {display_dict[result]}")
        if result:
            sub_set[:, :, 2] = sub_set[:, :, 2] / 2
        else:
            sub_set[:, :, 1] = sub_set[:, :, 1] / 2
        out_image[i * PATCH_SIZE:(i + 1) * PATCH_SIZE, j * PATCH_SIZE:(j + 1) * PATCH_SIZE, :] = sub_set
file_name = TARGET_IMAGE.split(".")
new_name = file_name[0]+"_Marked."+file_name[1]
cv2.imwrite(new_name, out_image)
print("Testing Complete, image saved")
