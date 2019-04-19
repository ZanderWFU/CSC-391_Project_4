import cv2
import numpy as np

TEST_MODE = False #disables saving

target_name = "DSC08606.JPG"
PATCH_SIZE = 200
palm_start = 185
non_palm_start = 1151
skip_start = 0

start_image = cv2.imread(target_name)
length = start_image.shape[0]
width = start_image.shape[1]
l_max = int(length/PATCH_SIZE)
w_max = int(width/PATCH_SIZE)

print(f"Testing {w_max * l_max} patches.")
im_count = 0
p_count = palm_start
n_count = non_palm_start
i_count = skip_start


for i in range (0, l_max - 1):
    for j in range (0, w_max - 1):
        sub_set = start_image[i*PATCH_SIZE:(i+1)*PATCH_SIZE, j*PATCH_SIZE:(j+1)*PATCH_SIZE, :]
        im_count = im_count + 1
        cv2.imshow(f"{im_count}: Palm or Not Palm?", sub_set)
        c = cv2.waitKey(0)
        if c == ord('a'):
            p_count = p_count + 1
            print(f"Image {im_count} saved as Palm_{p_count}.jpg")
            if not TEST_MODE:
                cv2.imwrite(f"Palm/Palm_{p_count}.jpg", sub_set)
        elif c == ord('s'):
            i_count = i_count + 1
            print(f"Image {im_count} saved as Skip_{i_count}.jpg")
            if not TEST_MODE:
                cv2.imwrite(f"Skip/Skip_{i_count}.jpg", sub_set)
        elif c == ord('d'):
            n_count = n_count + 1
            print(f"Image {im_count} saved as NotPalm_{n_count}.jpg")
            if not TEST_MODE:
                cv2.imwrite(f"NotPalm/NotPalm_{n_count}.jpg", sub_set)
        cv2.destroyAllWindows()


