import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def do_stuff(path, path2):
    us_image = np.array(nib.load(path).get_fdata())
    us_image = np.transpose(us_image)
    seg = nib.load(path2).get_fdata()
    seg = np.transpose(seg)
    x_size = us_image.shape[1]
    y_size = us_image.shape[0]
    seg_padded = np.pad(seg, 1)
    seg_padded = seg_padded.astype('uint8')
    us_image = us_image.astype('uint8')
    f = plt.figure(figsize=(10, 10))
    plt.imshow(us_image, cmap='gray')
    plt.imshow(np.ma.masked_where(seg == 0, seg), alpha=0.25, vmin=1)
    plot_name = "curr_subject + '_' + file_name"
    plt.show()
