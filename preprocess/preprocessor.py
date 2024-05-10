import os
import numpy as np
import gzip
import nibabel as nib
import glob
import shutil


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


def unzip(data_path):
    dataset_zipped = data_path + "/database_nifti"
    dataset_unzipped = data_path + "/database"
    os.mkdir(dataset_unzipped)
    for subject in os.listdir(dataset_zipped):
        unzip_subject(os.path.join(dataset_zipped, subject), dataset_unzipped)

def unzip_subject(path, dest_path):
    file_list = os.listdir(path)
    subject = path[-11:]
    dest_subject_path = dest_path+"/"+subject
    os.mkdir(dest_subject_path)
    for file in file_list:
        print(f"unzipping patient: {subject}")
        if file.endswith('.gz'):
            with gzip.open(path + "/" + file, 'rb') as zipped:
                with open(dest_subject_path + "/" + file[:-3], 'wb') as unzipped:
                    shutil.copyfileobj(zipped, unzipped)
