#from mpl_toolkits.basemap import Basemap
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from numpy import nan as NA
from PIL import Image
import os
from osgeo import gdal, osr
import numpy as np


def write_img(filename,im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
#判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
        #创建文件
    driver = gdal.GetDriverByName("GTiff")#数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data) #写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

if __name__=="__main__":
    write_img(filename, data)