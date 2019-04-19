import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split

from skimage.feature import local_binary_pattern

#Settings
OLD_SAMPLE_DATA = False #Use sample data from project 3 for training
PALM_TEST_SIZE = 0.25 #proportion of the palm data to be used in the test set (not training)
NOTPALM_TEST_SIZE = 0.75 #proportion of the NotPalm data to be used in the test set (not training)
#-LBP Settings
radius = 3
n_points = 8 * radius
METHOD = 'uniform'

#Name: Zander Miller
#CSC-391: Project 4

#Getting the images
print("Identifying Palm Images...")
Palm_list = glob("Sample_Data/Palm/*")
print("Identifying NotPalm Images...")
NotPalm_list = glob("Sample_Data/NotPalm/*")

Palm_refs_train = []
Palm_refs_test = []
NotPalm_refs_train = []
NotPalm_refs_test = []

if OLD_SAMPLE_DATA:
    print("Training Palm References...")
    ptr_list = glob("Sample_Data_Old/Palm/*")
    for file in ptr_list:
        infile = cv2.imread(file, 0)
        Palm_refs_train.append(local_binary_pattern(infile, n_points, radius, METHOD))
    print("Training NotPalm References...")
    nptr_list = glob("Sample_Data_Old/NotPalm/*")
    for file in nptr_list:
        infile = cv2.imread(file, 0)
        NotPalm_refs_train.append(local_binary_pattern(infile, n_points, radius, METHOD))
    print("Prepping Palm Test References...")
    for file in Palm_list:
        infile = cv2.imread(file, 0)
        Palm_refs_test.append(local_binary_pattern(infile, n_points, radius, METHOD))
    print("Prepping NotPalm Test References...")
    for file in NotPalm_list:
        infile = cv2.imread(file, 0)
        NotPalm_refs_test.append(local_binary_pattern(infile, n_points, radius, METHOD))
else:
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
    print("Prepping Palm Test References...")
    for file in pte_list:
        infile = cv2.imread(file, 0)
        Palm_refs_test.append(local_binary_pattern(infile, n_points, radius, METHOD))
    print("Prepping NotPalm Test References...")
    for file in npte_list:
        infile = cv2.imread(file, 0)
        NotPalm_refs_test.append(local_binary_pattern(infile, n_points, radius, METHOD))


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

palm_matches = 0
palm_misses = 0
palm_total = len(Palm_refs_test)
notpalm_matches = 0
notpalm_misses = 0
notpalm_total = len(NotPalm_refs_test)
i = 1
print("Beginning Image Testing...")
for reference in Palm_refs_test:
    result = matching(Palm_refs_train, NotPalm_refs_train, reference)
    if result:
        palm_matches += 1
        print(f"Palm_{i} was correctly identified")
    else:
        palm_misses += 1
        print(f"Palm_{i} was missed")
    i += 1
i = 1
for reference in NotPalm_refs_test:
    result = matching(Palm_refs_train, NotPalm_refs_train, reference)
    if result:
        notpalm_misses += 1
        print(f"NotPalm_{i} had a false positive")
    else:
        notpalm_matches += 1
        print(f"NotPalm_{i} was correctly identified")
    i += 1
print("Image Testing Complete!")
print(f"Palm Images:\nHits: {palm_matches}, {int((palm_matches / palm_total)* 100)}% \tMisses: {palm_misses}, "
      f"{int((palm_misses / palm_total) * 100)}% \tTest Set Size: {palm_total}")
print(f"NotPalm Images:\nHits: {notpalm_matches}, {int((notpalm_matches / notpalm_total)* 100)}% \tMisses: {notpalm_misses}, "
      f"{int((notpalm_misses / notpalm_total) * 100)}% \tTest Set Size: {notpalm_total}")