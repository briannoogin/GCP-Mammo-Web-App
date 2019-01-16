import pydicom
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from pydicom.data import get_testdata_files
import pandas
import os
from os import path
import glob 
import pickle
import json
# module is used to import dicom images and write them to jpgs for training and testing 

# function reads in dicom image and returns the pixel array of the image
# has option with display_image parameter to display the image
def display_image(file_path,display_image):
    ds = pydicom.dcmread(file_path)
    if display_image:
        plt.imshow(color.rgb2gray(ds.pixel_array),cmap=plt.gray())
        plt.show()
    return ds.pixel_array

# function converts csv to pandas dataframe object
def read_csv_descriptions(csv_path):
    csv_df = pandas.read_csv(csv_path,index_col='patient_id')
    return csv_df

# converts all dicom images to jpegs
def convert_dcm2jpg():
    save_image = True
    # get list of all dicom images
    path_list = get_dicom_folder()
    # list of classes for all patient cases
    pathology_list = []
    img_list = []
    df = pandas.read_csv('mass_case_description_test_set.csv')

    # transform string classes into integer numbers
    # 0: benign
    # 1: benign without callback
    # 2: malignant
    labels,key = pandas.factorize(df['pathology'],sort=True)
    for idx,file_path in enumerate(path_list):
        img = display_image(file_path,False)
        file_name = 'data/test/' + df['pathology'][idx] + '/' + df['patient_id'][idx] + '_' + df['left or right breast'][idx] + '_' + df['image view'][idx] + '_' + str(df['breast_density'][idx]) + '_' + str(df['abnormality id'][idx])
        if save_image:
            img = io.imsave(file_name + ".jpg",img)
        pathology_list.append(int(labels[idx]))
        img_list.append(file_name)
    assert(len(pathology_list) == len(img_list))
        
# gets path of the dicom image folder
def get_dicom_folder():
    # get working directory
    cwd = os.getcwd();
    # remove the current directory from the path 
    base = os.path.join(cwd,'CBIS-DDSM')
    # image path is within the nested folders
    path = os.path.join(base,'test','*','*','*','*')
    path_list = []
    # get matching pattern with glob
    for name in glob.glob(path,recursive=True):
        path_list.append(name)
    return sorted(path_list)

# delete all region of interest photos 
def delete_roi_mask():
    path_list = get_dicom_folder()
    for file in path_list:
        if os.path.getsize(file) > 5000 * 1024:
            os.remove(file)

if __name__ == "__main__":
    #delete_roi_mask()
   convert_dcm2jpg()