
"""
2022/10/12
xm
提取缺失时刻名称，并根据前后两个时刻通过时间插值生成缺失时刻的影像

理想情况下每个文件下有144张影像

"""


import os
import numpy as np
from time_inter import time_inter

fpath ="H:/Himawari-8/2018/"
path = os.listdir(fpath)
print("一级文件：", path)
print("共读取%d个文件" % len(path))

savepath = "E:/H08/GPM-V2/time_inter0601/"

file = []
all_count = 0

def name(h08path):
    rsname = os.path.split(h08path)[1]
    fname = rsname[7:20]
    return fname

for i in range(3): #len(path)
    imgpath = os.path.join(fpath, path[i]) #"H:/Himawari-8/2018/20180601/"
    fname = path[i]
    print('年_月_日：', fname) #20180601
    ten_img = sorted(os.listdir(imgpath) )#一天 142张影像
    print(ten_img)
    print("读取影像：", len(ten_img))

    for j in range(len(ten_img)-1):#len(img)
        path1 = os.path.join(imgpath, ten_img[j])
        path2 = os.path.join(imgpath, ten_img[j+1])
        #print(path1)
        t1name = name(path1)
        t2name = name(path2)
        print(t1name)
        t1 = int(t1name.split("_")[1])
        t2 = int(t2name.split("_")[1])
        interval = t2 - t1
        print(interval)
        if interval == 10 or interval == 50:
            pass
        else:
            time_inter(path1, path2, savepath, obj=2)

"""

fpath ="H:/Himawari-8/2018/"
path = os.listdir(fpath)
print("一级文件：", path)
print("共读取%d个文件" % len(path))

file = []
all_count = 0

for i in range(1): #len(path)
    imgpath = os.path.join(fpath, path[i]) #"H:/Himawari-8/2018/20180601/"
    fname = path[i]
    print('年_月_日：', fname) #20180601
    ten_img = os.listdir(imgpath) #一天 142张影像
    print(ten_img)
    print("读取影像：", len(ten_img))

    all_time = [] #存储遍历文件后的时间名称
    for j in range(len(ten_img)):#len(img)
        datapath = os.path.join(imgpath, ten_img[j]) #ten_img[j] 每个时刻影像的名称
        #print("影像所在路径：", datapath)  #H:/Himawari-8/2018/20180601/NC_H08_20180602_1330_R21_FLDK.06001_06001.nc
        file.append(ten_img[j])

        rsname = os.path.split(datapath)[1]
        name = os.path.splitext(ten_img[j])  # 区分文件名 和后缀，即('NC_H08_20180715_1030_R21_FLDK.06001_06001', '.nc')
        name1 = name[0].split("_")  # 'NC', 'H08', '20180715', '1030', 'R21', 'FLDK.06001', '06001']
        time = name1[2] + "_" + name1[3]
        print("time", time)
        all_time.append(time)

        cal_time = [] #存储缺失时刻的 名称
        for m in range(len(all_time) - 1):
            t1 = int(all_time[m].split("_")[1])
            t2 = int(all_time[m + 1].split("_")[1])
            interval = t2 - t1
            if interval == 10 or interval == 50:
                pass
            else:
                cal_time.append(t1)
        print(cal_time)

"""

