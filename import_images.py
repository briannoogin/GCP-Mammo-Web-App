import pydicom
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
from pydicom.data import get_testdata_files
def display_image(file_path):
    ds = pydicom.dcmread(file_path)
    print(ds.pixel_array.shape)
    plt.imshow(color.rgb2gray(ds.pixel_array),cmap=plt.gray())
    plt.show()
    return ds.pixel_array
def convert_dcm2jpg(file_path):
    img = display_image(file_path)
    io.imsave("breast1.jpg",img)
convert_dcm2jpg("000001.dcm")
