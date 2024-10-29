import os
import numpy as np
import gzip
import nibabel as nib
import glob
import shutil
import cv2

from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio


def check_dataset_integrity(data_path, print_results=False):
    log_entry = ""
    dataset_path = data_path + "/database_nifti"
    files = open("preprocess/integrity_check.txt", "r").read().split(",")
    for subject in os.listdir(dataset_path):
        log_entry += check_subject_integrity(os.path.join(dataset_path, subject), files)
    logfile = open(data_path + "/integrity_log.txt", 'w+')
    logfile.write(log_entry)
    if print_results:
        print(log_entry)
    logfile.close()


def check_subject_integrity(path: os.path, files):
    file_list = os.listdir(path)
    file_list.sort()
    long_entry = ""
    subject = path[-11:]
    for i in range(3):
        if files[i] not in file_list:
            long_entry += f"{subject} missing file {files[i]} "
    for i in range(3, 15):
        if subject+files[i] not in file_list:
            long_entry += f"{subject} missing file {subject + files[i]} "
    if long_entry == "":
        long_entry = f"{subject} successfully checked for file integrity "
    return long_entry + ";\n"


def unzip(data_path, print_progress=False):
    dataset_zipped = data_path + "/database_nifti"
    dataset_unzipped = data_path + "/database"
    os.mkdir(dataset_unzipped)
    for i, subject in enumerate(os.listdir(dataset_zipped)):
        unzip_subject(os.path.join(dataset_zipped, subject), dataset_unzipped)
        if print_progress:
            print(f"done {i} out of 500")

def unzip_subject(path, dest_path):
    file_list = os.listdir(path)
    subject = path[-11:]
    dest_subject_path = dest_path+"/"+subject
    os.mkdir(dest_subject_path)
    for file in file_list:
        if file.endswith('.gz'):
            with gzip.open(path + "/" + file, 'rb') as zipped:
                with open(dest_subject_path + "/" + file[:-3], 'wb') as unzipped:
                    shutil.copyfileobj(zipped, unzipped)


def copy_configs(src_path, dest_path):
    for subject in os.listdir(src_path):
        subject_src_path = f"{src_path}/{subject}"
        src1 = open(subject_src_path + '/Info_2CH.cfg', 'rb')
        src2 = open(subject_src_path + '/Info_4CH.cfg', 'rb')
        subject_dest_path = f"{dest_path}/{subject}"
        dest1 = open(subject_dest_path + '/Info_2CH.cfg', 'wb')
        dest2 = open(subject_dest_path + '/Info_4CH.cfg', 'wb')
        shutil.copyfileobj(src1, dest1)
        shutil.copyfileobj(src2, dest2)

def convert_subject_to_jpg(path, dest_path, base_transform):
    file_list = os.listdir(path)
    subject = path[-11:]
    dest_subject_path = dest_path + "/" + subject
    os.mkdir(dest_subject_path)
    # only do 4ch ed and es for now to see if there will be any memory improvements
    for file in file_list:
        if "4CH_ED" in file or "4CH_ES" in file:
            if "gt" in file:
                continue
            file_path = path + "/" + file
            img = nib.load(file_path)
            img = np.array(np.transpose(img.get_fdata()))
            img = base_transform(image=img)
            final_path = dest_subject_path + "/" + file[:-3] + "jpg"
            cv2.imwrite(final_path, img["image"])


def convert_dataset_to_jpg(source_path, dest_path, base_transform):
    os.mkdir(dest_path)
    for subject in os.listdir(source_path):
        convert_subject_to_jpg(os.path.join(source_path, subject), dest_path, base_transform)


def expand_series(source_path, dest_path, transform_method=None, param=None):
    os.mkdir(dest_path)
    subject_list = os.listdir(source_path)
    for subject in subject_list:
        working_dir = f"{source_path}/{subject}"
        dest = f"{dest_path}/{subject}"
        os.mkdir(dest)
        series = f"{working_dir}/{subject}_4CH_half_sequence.nii"
        images = np.array(np.transpose(nib.load(series).get_fdata()))
        for i in range(images.shape[0]):
            if transform_method is not None:
                img = wavelet_denoise(images[i, :], transform_method, param)
                cv2.imwrite(f"{dest}/4CH_{i}_.jpg", img)
            else:
                cv2.imwrite(f"{dest}/4CH_{i}_.jpg", images[i, :])

        series = f"{working_dir}/{subject}_4CH_half_sequence_gt.nii"
        images = np.array(np.transpose(nib.load(series).get_fdata()))
        for i in range(images.shape[0]):
            img = images[i, :] * 85
            cv2.imwrite(f"{dest}/4CH_{i}_gt.png", img)


def wavelet_denoise(img, method, param):
    img = np.stack((img,) * 3, axis=-1).astype(np.uint8)
    noisy = img_as_float(img)
    sigma_est = estimate_sigma(noisy, channel_axis=-1, average_sigmas=True)
    if method == 'BayesShrink':
        transformed = denoise_wavelet(
            noisy,
            channel_axis=-1,
            convert2ycbcr=True,
            method='BayesShrink',
            mode='soft',
            rescale_sigma=True,
        )
    else:
        transformed = denoise_wavelet(
            noisy,
            channel_axis=-1,
            convert2ycbcr=True,
            method='VisuShrink',
            mode='soft',
            sigma=sigma_est * param,
            rescale_sigma=True,
        )
    return (transformed[:, :, 0] * 255).astype(np.float64)
