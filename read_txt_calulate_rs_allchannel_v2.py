"""
2022/10/8
xm
目的：
（1）将每个时刻的目标框，再根据前两个时刻都大于某个值，并且均有相同的目标框，进行最终的保留
（2）根据GPM和对流云追踪后保留的框，在三通道（h08影像的通道8,13,16）的上进行目标框内的像素计算（取将每个通道的像素值从低到高排序，然后取其25%，求平均）
（3）将计算后的数据写入excel，当做样本集

"""
import numpy as np
import math
import os
import netCDF4 as nc
from netCDF4 import Dataset
import cv2
import gdal
import matplotlib.pyplot as plt

def read_allchannel_tif(filename): #文件路径
    rsname = os.path.split(filename)[1]
    name = os.path.splitext(rsname)  # 区分文件名 和后缀，即('NC_H08_20180715_1030_R21_FLDK.06001_06001', '.nc')
    name1 = name[0].split("_")  # 'NC', 'H08', '20180715', '1030', 'R21', 'FLDK.06001', '06001']
    time = name1[2] + "_" + name1[3]

    dataset = gdal.Open(filename)
    if dataset == None:
        print(filename + "无法打开")
        return
    im_width = dataset.RasterXSize                              # 栅格矩阵的列数（宽）
    im_height = dataset.RasterYSize                             # 栅格矩阵的行数（高）
    im_bands = dataset.RasterCount                              # 波段数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)    # 获取数据    (16, 1801, 2401)
    im_geotrans = dataset.GetGeoTransform()                     # 获取仿射矩阵信息
    im_proj = dataset.GetProjection()                           # 获取投影信息
    im_data = im_data.transpose((1, 2, 0))[1340:1681, 980:1481]# (1801, 2401, 16)-->(341, 501, 16)  从中国地区裁剪成华南地区
    return im_data, time
#
# data_path = "Z:/A-SYY-DATA-h08/chinal/2018/20180601/NC_H08_20180601_0000_R21_FLDK.06001_06001.tif"
# data = read_allchannel_tif(data_path)[0]
# time = read_allchannel_tif(data_path)[1]
# print("time:", time)
# print(data.shape)
# print("-----------------------------------")
# plt.imshow(data[:, :, 12])
# plt.show()

def yz_allchannel(data_allchannel):
    yz = np.zeros((341, 501, 16))  # 后续可以替换掉
    for m in range(data_allchannel.shape[0]):
        for n in range(data_allchannel.shape[1]):
            if 0 < data_allchannel[m, n, 12] < 273:
                yz[m, n, :] = data_allchannel[m, n, :]
            else:
                yz[m, n, :] = 0
    return yz

def c1_c6_mask(data_c13):
     #阈值后的c13
    mask = np.ones_like(data_c13)
    for i in range(data_c13.shape[0]):
        for j in range(data_c13.shape[1]):
            if data_c13[i, j] == 0:
                mask[i, j] = 0
            else:
                mask[i, j] = 1
    return mask

"""
将每个通道的像素值从低到高排序，然后取其25%，求平均，将其作为云团的像素值（采取方式：将其转成list ,去除0值，然后排序，截取）
主要计算出t1_c16_mean 、t1_c13_mean 、t1_c08_mean 
"""
#按顺序排序，求取均值，针对红外通道进行计算
def sort_mean(data):
    f2 = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 0:
                pass
            else:
                f2.append(data[i, j])
    f2_sort = sorted(f2) #从小到大排序
    count = len(f2_sort)
    count = math.ceil(count*0.25)
    f2_sort_clip = f2_sort[0: count]
    f2_end = f2_sort_clip[-1]
    f2_sort_clip_average = np.mean(f2_sort_clip)-273 #求取平均亮温
    return f2_sort_clip_average, f2_end

#针对可见光和近红外进行计算（c1-6）
def sort_mean_c1_c6(data):
    f2 = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 0:
                pass
            else:
                f2.append(data[i, j])
    f2_sort = sorted(f2) #从小到大排序
    count = len(f2_sort)
    count = math.ceil(count*0.25)
    f2_sort_clip = f2_sort[0: count]
    f2_sort_clip_average = np.mean(f2_sort_clip)
    return f2_sort_clip_average


"""
根据第13通道，计算前25%的像素的平均值，并计算16和8通道的平均亮温
根据跟踪后的云目标框，计算典型子集
"""

def one_channel(t1_c01, t1_c13):
    t1_c01_all = []
    t1_c13_mean = sort_mean(t1_c13)  # 按照第13通道进行计算
    for i in range(t1_c13.shape[0]):
        for j in range(t1_c13.shape[1]):
            if t1_c13[i][j] == 0:
                pass
            elif t1_c13[i][j] <= t1_c13_mean[1]:#t1_c13_mean[1]是一个数值
                t1_c01_all.append(t1_c01[i][j])
    t1_c01_mean = np.mean(t1_c01_all) - 273  #红外通道 亮温
    return t1_c01_mean

#要先有目标框，然后才能计算均值

def classical_subset_allchannel(data): #data (,,16)
    t1_c13 = data[:, :, 12]
    t1_c13_mean = sort_mean(t1_c13)
    data_mean = []
    for i in range(data.shape[2]):
        if i < 6:
            mask = c1_c6_mask(t1_c13)
            c = data[:, :, i]*mask
            data_mean.append(sort_mean_c1_c6(c))
        elif i == 12:
            data_mean.append(t1_c13_mean[0])
        else:
            c = data[:, :, i]
            c_mean = one_channel(c, t1_c13)
            data_mean.append(c_mean)
    return data_mean



import xlwt
book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('ci', cell_overwrite_ok=True)
col = ('time', 'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'label')  # 列表名称
savepath = 'E:/H08/tree/data/allchannel/ci-allchannel-201806-t2-try.xls'

#读取txt格式
path = "E:/H08/tree/data/"
mode = "t2"  # 第一时刻 or 第二时刻
f = open(path + "try.txt", encoding="utf-8")  # 读取的是每个时刻的目标框  201806_GPM,201807_GPM,201808_GPM
line = f.readline()
data = []
while line:
    a = line.split("\n")  # 默认按空格划分
    data.append(a)
    line = f.readline()
f.close()
#先批量处理试试

for i in range(1, 2):#len(data)
    print("----i:", i)
    time = data[i][0][2:15]  # 提取时间
    print("time:", time)
    p = "Z:/A-SYY-DATA-h08/chinal/2018/" #此路径下保存了中国地区 H08所有通道的信息
    fpath = data[i][0][2:10]
    pathname = "NC_H08_" + time + "_R21_FLDK.06001_06001.tif"
    objpath = os.path.join(p, fpath, pathname)
    print(objpath)
    # print(fpath)
    # print(pathname)
    t1 = eval(data[i][0][17:-1])  # 所有目标框信息,列表形式
    print(t1)

    data2 = read_allchannel_tif(objpath)[0]
    data3 = yz_allchannel(data2) #阈值处理

    c13 = data2[:,:,12]

    # 提取目标框
    x = t1[0]
    y = t1[1]
    w = t1[2]
    h = t1[3]
    gpm = t1[-1]
    obj_data = data3[y:y + h, x:x + w] #再阈值的基础上，进行裁剪
    # img_rectangle_th = cv2.rectangle(data2, (x, y), (x + w, y + h), (255, 0, 0), 1)  # 截取出感兴趣的区域
    # cv2.imwrite("E:/H08/GPM-V2/"+time+".tif", img_rectangle_th)
    print("目标框的形状大小：", obj_data.shape)
    # c16 = obj_data[:, :, 15]
    # c13 = obj_data[:, :, 12]
    # c1 = obj_data[:, :, 0]
    # c2 = obj_data[:, :, 1]
    # c3 = obj_data[:, :, 2]
    # c4 = obj_data[:, :, 3]
    # c5 = obj_data[:, :, 4]
    # c6 = obj_data[:, :, 5]
    # c7 = obj_data[:, :, 6]
    # c8 = obj_data[:, :, 7]

    out = classical_subset_allchannel(obj_data)
    c1_mean = out[0]
    c2_mean = out[1]
    c3_mean = out[2]
    c4_mean = out[3]
    c5_mean = out[4]
    c6_mean = out[5]
    c7_mean = out[6]
    c8_mean = out[7]
    c9_mean = out[8]
    c10_mean = out[9]
    c11_mean = out[10]
    c12_mean = out[11]
    c13_mean = out[12]
    c14_mean = out[13]
    c15_mean = out[14]
    c16_mean = out[15]
    final_data = []  # 将计算的数据放入此

    if mode == "t2":
        final_data.append((time, c1_mean, c2_mean, c3_mean, c4_mean, c5_mean, c6_mean, c7_mean, c8_mean,c9_mean, c10_mean, c11_mean, c12_mean, c13_mean,
                       c14_mean, c15_mean, c16_mean, gpm))

    else:
        final_data.append((time, c1_mean, c2_mean, c3_mean, c4_mean, c5_mean, c6_mean, c7_mean, c8_mean, c9_mean, c10_mean,
                       c11_mean, c12_mean, c13_mean,
                       c14_mean, c15_mean, c16_mean))

    for m in range(0, len(col)):
        sheet.write(0, m, col[m])
    print("final_data", final_data)

    datalist = list(final_data)[0]
    print("datalist:", datalist)
    for j in range(0, len(datalist)):
        sheet.write(i + 1, j, str(datalist[j])) #i 为行， j为列
    book.save(savepath)
