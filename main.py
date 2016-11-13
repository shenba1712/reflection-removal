import cv2
import time
import averaging
import rpca
import frpca
import util
import numpy as np
from sklearn.metrics import mean_squared_error

BACKGROUND_FOLDER = "../data/background/"
GROUND_TRUTH_FOLDER = "../data/ground_truth/"
GROUND_TRUTH_IMAGE = cv2.imread("../data/ground_truth/truth.png", cv2.IMREAD_UNCHANGED)

SET1_FOLDER = "../data/set1/"
SET2_FOLDER = "../data/set2/"
SET3_FOLDER = "../data/set3/"
SET4_FOLDER = "../data/set4/"

set1_ext = [10, 30, 50, 70, 90]
set2_ext = [50, 40, 30]
set3_ext = [2, 4, 6, 8, 10]
set4_ext = [4, 8, 12, 16, 20]

set2_fixed_r = [2, 2, 2]


def write_image(path_format, images, names):
    print path_format
    print len(images)
    print images[0].shape
    image_length = len(images)
    if image_length == 1:
        cv2.imwrite(path_format + ".png", images[0])
    elif image_length > 1:
        for i in xrange(image_length):
            cv2.imwrite(path_format + "_" + names[i] + ".png", images[i])

# # averaging
# print "SET1 AVERAGING"
# for ext in set1_ext:
#     folder = "set1_" + str(ext) + "/"
#     print "folder: ", folder
#     path = SET1_FOLDER + folder
#     start = time.time()
#     background = averaging.get_background(path)
#     print "time: ", time.time() - start
#     cv2.imwrite(BACKGROUND_FOLDER + "averaging_set1_" + str(ext) + ".png", background)
#     print "error: ", mean_squared_error(GROUND_TRUTH_IMAGE.ravel(), background.ravel())
#     print
#
# # rpca
# print "SET1 RPCA"
# for ext in set1_ext:
#     folder = "set1_" + str(ext) + "/"
#     print "folder: ", folder
#     path = SET1_FOLDER + folder
#     start = time.time()
#     background, names = rpca.get_background(path)
#     print "time: ", time.time() - start
#     write_image(BACKGROUND_FOLDER + "rpca_set1_" + str(ext), background, names)
#
#     print "error: ", util.compute_mean_square_error(background, GROUND_TRUTH_IMAGE)
#     print "bg shape: ", np.array(background).shape
#     print
#
# # averaging
# print "SET2 AVERAGING"
# for ext in set2_ext:
#     folder = "set2_" + str(ext) + "/"
#     print "folder: ", folder
#     path = SET2_FOLDER + folder
#     start = time.time()
#     background = averaging.get_background(path)
#     print "time: ", time.time() - start
#     cv2.imwrite(BACKGROUND_FOLDER + "averaging_set2_" + str(ext) + ".png", background)
#     print "error: ", mean_squared_error(GROUND_TRUTH_IMAGE.ravel(), background.ravel())
#     print
#
# # rpca
# print "SET2 RPCA"
# for ext in set2_ext:
#     folder = "set2_" + str(ext) + "/"
#     print "folder: ", folder
#     path = SET2_FOLDER + folder
#     start = time.time()
#     background, names = rpca.get_background(path)
#     print "time: ", time.time() - start
#     write_image(BACKGROUND_FOLDER + "rpca_set2_" + str(ext), background, names)
#
#     print "error: ", util.compute_mean_square_error(background, GROUND_TRUTH_IMAGE)
#     print "bg shape: ", np.array(background).shape
#     print
#
# frpca
print "SET2 FRPCA"
for i in xrange(len(set2_ext)):
    ext = set2_ext[i]
    r = set2_fixed_r[i]
    folder = "set2_" + str(ext) + "/"
    print "folder: ", folder
    path = SET2_FOLDER + folder
    start = time.time()
    background = frpca.get_background(path, r=r)
    print "time: ", time.time() - start
    cv2.imwrite(BACKGROUND_FOLDER + "frpca_set2_" + str(ext) + ".png", background)
    print "error: ", mean_squared_error(GROUND_TRUTH_IMAGE.ravel(), background.ravel())
    print "bg shape: ", background.shape
    print
#
# # averaging
# print "SET3 AVERAGING"
# for ext in set3_ext:
#     folder = "set3_" + str(ext) + "/"
#     print "folder: ", folder
#     path = SET3_FOLDER + folder
#     start = time.time()
#     background = averaging.get_background(path)
#     print "time: ", time.time() - start
#     cv2.imwrite(BACKGROUND_FOLDER + "averaging_set3_" + str(ext) + ".png", background)
#     print "error: ", mean_squared_error(GROUND_TRUTH_IMAGE.ravel(), background.ravel())
#     print
#
# # rpca
# print "SET3 RPCA"
# for ext in set3_ext:
#     folder = "set3_" + str(ext) + "/"
#     print "folder: ", folder
#     path = SET3_FOLDER + folder
#     start = time.time()
#     background, names = rpca.get_background(path)
#     print "time: ", time.time() - start
#     write_image(BACKGROUND_FOLDER + "rpca_set3_" + str(ext), background, names)
#
#     print "error: ", util.compute_mean_square_error(background, GROUND_TRUTH_IMAGE)
#     print "bg shape: ", np.array(background).shape
#     print
#
# # averaging
# print "SET4 AVERAGING"
# for ext in set4_ext:
#     folder = "set4_" + str(ext) + "/"
#     print "folder: ", folder
#     path = SET4_FOLDER + folder
#     start = time.time()
#     background = averaging.get_background(path)
#     print "time: ", time.time() - start
#     cv2.imwrite(BACKGROUND_FOLDER + "averaging_set4_" + str(ext) + ".png", background)
#     print "error: ", mean_squared_error(GROUND_TRUTH_IMAGE.ravel(), background.ravel())
#     print
#
# # rpca
# print "SET4 RPCA"
# for ext in set4_ext:
#     folder = "set4_" + str(ext) + "/"
#     print "folder: ", folder
#     path = SET4_FOLDER + folder
#     start = time.time()
#     background, names = rpca.get_background(path)
#     print "time: ", time.time() - start
#     write_image(BACKGROUND_FOLDER + "rpca_set4_" + str(ext), background, names)
#
#     print "error: ", util.compute_mean_square_error(background, GROUND_TRUTH_IMAGE)
#     print "bg shape: ", np.array(background).shape
#     print
