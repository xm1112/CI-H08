
"""
将.nc数据转换成tif
"""
import os
import numpy as np
# from time_inter import time_inter
from hansu import read_h08, write_img, read_h08_allchannel

fpath ="H:/Himawari-8/2018/"
path = sorted(os.listdir(fpath))
print("一级文件：", path)
print("共读取%d个文件" % len(path))

#savepath = "H:/Himawari-8/alltif/2018"
# savepath = "I:/Himawari-8/c/2018/"

savepath = "C:/Users/DELL/Desktop/h08/"

file = []
all_count = 0

def name(h08path):
    rsname = os.path.split(h08path)[1]
    fname = rsname[7:20]
    return fname

for i in range(30, 31): #len(path)
    imgpath = os.path.join(fpath, path[i]) #"H:/Himawari-8/2018/20180601/"
    fname = path[i]
    print('年_月_日：', fname) #20180601
    ten_img = sorted(os.listdir(imgpath))#一天 142张影像
    print(ten_img)

    final_path = savepath+"/"+fname
    print(final_path)
    if os.path.exists(final_path) ==True:
        pass
    else:
        final_path = os.makedirs(savepath + "/" + fname)

    print("读取影像：", len(ten_img))

    for j in range(7, 8):#len(ten_img)
        path1 = os.path.join(imgpath, ten_img[j])
        print(path1)
        t1name = os.path.splitext(ten_img[j])[0]
        GS = os.path.splitext(ten_img[j])[1]
        print("格式：", GS)
        if GS ==".nc":
            print(t1name)
            img =read_h08_allchannel(path1)[0] #(w,h,16)
            #转置
            img1 = img.transpose((2, 0, 1))
            print(img1.shape)
            img1 = write_img(savepath+"/"+fname+"/" + str(t1name)+".tif", img1)
        else:
            pass
