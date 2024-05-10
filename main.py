import cv2
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import skimage
from preprocess.preprocessor import*
from tools.debug import*

# img = nib.load("data/database/patient0001/patient0001_4CH_ES.nii")
# nii_data = np.array(np.transpose(img.get_fdata()), dtype=np.uint8)
#
# cv2.imshow("img", nii_data)
# cv2.waitKey(0)
do_stuff("data/database/patient0001/patient0001_4CH_ES.nii", "data/database/patient0001/patient0001_4CH_ES_gt.nii")


