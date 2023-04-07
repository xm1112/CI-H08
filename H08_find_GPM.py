"""
wangyan
2022/10/19
根据葵花八数据 在GPM上查找最临近时刻的  指定区域 降水最大值
将降水数据进行归类
  0     :0
<2.5    :1
2.5~16  :2
>16     :3

"""
import numpy as np
import math
import os
import h5py
import time
import datetime
from datetime import datetime,timedelta,date
import xlwt

# 最邻近插值
def BiLinear_interpolation(img,dstH,dstW):
    scrH,scrW= img.shape
    img = np.pad(img,((0,1),(0,1)),'constant')  # 将数组按指定方法填充
    retimg = np.zeros((dstH,dstW))
    for i in range(dstH):
        for j in range(dstW):
            scrx = (i+1)*(scrH/dstH)-1
            scry = (j+1)*(scrW/dstW)-1
            x=math.floor(scrx)
            y=math.floor(scry)
            u = scrx - x
            v = scry - y
            retimg[i,j] = (1-u)*(1-v)*img[x,y] + u*(1-v)*img[x+1,y] + (1-u)*v*img[x,y+1] + u*v*img[x+1,y+1]
    return retimg

def read_GPM(file):
    # 读取GPM数据
    domain = os.path.abspath(r"D:/GPM/2018/08")
    fpath = os.path.join(domain, file)  # 文件的完整名称
    dataset = h5py.File(fpath, "r")
    precip = dataset['Grid/precipitationCal'][:]
    # print(precip)
    precip = precip.reshape(3600, 1800)  # reshape() 将数据按既定纬度进行整理
    precip = np.transpose(precip)  # 1800*3600
    data = precip[1079:1166, 2844:2971]
    data = BiLinear_interpolation(data, data.shape[0] * 4, data.shape[1] * 4)  # 插值为0.025°
    data = data[5:346, 5:506]  # 341*501   # 范围为
    data = data[::-1]
    # print(data)
    return data

def read_GPM1(file):
    # 读取GPM数据
    domain = os.path.abspath(r"D:/GPM/2018/08")
    fpath = os.path.join(domain, file)  # 文件的完整名称
    dataset = h5py.File(fpath, "r")
    precip = dataset['Grid/precipitationCal'][:]
    # print(precip)
    precip = precip.reshape(3600, 1800)  # reshape() 将数据按既定纬度进行整理
    precip = np.transpose(precip)  # 1800*3600
    data = precip[1079:1166, 2844:2971]
    data = BiLinear_interpolation(data, data.shape[0] * 4, data.shape[1] * 4)  # 插值为0.025°
    data = data[5:346, 5:506]  # 341*501   # 范围为
    data = data[::-1]
    # print(data)
    return data

def area_max(data):
    # 求目标框区域的最大值
    p = 0
    max = 0
    arr = []  # 存储降水max的信息 [行，列，最大值]
    row = 0  # 行
    col = 0  # 列
    for a in range(yi):
        for b in range(xi):
            p = sr[a][b]
            if p > max:
                max = p
                row = y + a
                col = x + b
                arr = [row,col,max]
            # 最大值为0，则记录左上角的位置
            if max == 0:
                row = y
                col = x
                arr = [row, col, max]
    return arr

def group_GPM(max):
    # 对最大值进行分组
    # {0:0,0~2.5:1,2.5~16:2,>16:3}
    level = 0
    if 0 < max < 2.5:
        level = 1
    elif 2.5 <= max <= 16:
        level = 2
    elif max > 16:
        level = 3
    else:
        level = 0
    return level

Files = os.listdir("D:\\GPM\\2018\\07")
files = sorted(Files)
# print(files)


# 新建txt
# 写入日期及降水量最大值的分组信息
matching = open("D:/GPM/20180601_GPM.txt","w",encoding = "utf-8")
# 写入信息到txt



# 读取txt 文件
txt = "C:/Users/wangyan/Desktop/nnnn.txt"
f = open(txt,encoding = "utf-8")
# 读取文件中所有行
lines = f.readlines()
# print(len(lines))

# 存储
datalist0 = []
datalist1 = []   # 空数组 降水量最大值范围在 2.5~16
datalist2 = []   # 空数组 降水量最大值 > 16
datalist3 = []

for line in lines:
    result = []
    time = line[2:15]  # 时刻信息 20180602_0340
    # print(time)
    # 匹配时间（查找离H8数据时间最近的GPM图像
    i = int(time[6:8])  # int 类型   # 天
    # print(i)
    j = (time[9:11])    # 小时
    k = (time[11:13])   # 分钟
    # print(i,j,k)
    # 如果时刻 为00分，读取相对应图像
    """
    if k == 0:
        k += 1
    """
    t = 48*(int(i)-1) + int(j)*2 + math.ceil(int(k)/30)
    # print(t)

    # 读取GPM数据
    #data = read_GPM(files[t])
    data = read_GPM("3B-HHR.MS.MRG.3IMERG.20180801-S000000-E002959.0000.V06B.HDF5")
    # 读取目标框数据内降水量最大值
    xy = eval(line[17:-2])  # 可能是"list", 也可能是"tuple"
    #print(xy)   # [71,103,5,5,74,105]   list   [左上角(列,行),大小范围(列,行),区域内中心点(列,行)]
    x = xy[0]
    xi = xy[2]
    y = xy[1]
    yi = xy[3]
    # 框选目标区域
    sr = data[y:y + yi, x:x + xi]  # 确定目标框
    max_array1 = area_max(sr)      # 目标框区域最大值 [行,列,最大值]
    # print(max_array1)
    value1 = max_array1[2]
    # 如果最大值小于2.5，则找下一幅图
    max_array = []                      # 存储
    if value1 < 2.5:
        # data = read_GPM(files[t+1])    # 读取下一时刻的图像
        data = read_GPM("3B-HHR.MS.MRG.3IMERG.20180801-S003000-E005959.0030.V06B.HDF5")
        sr = data[y:y + yi, x:x + xi]  # 确定目标框
        max_array2 = area_max(sr)      # 调用 area_max函数 返回图片框选范围内的最大值 [行,列,最大值]
        value2 = max_array2[2]
        # 记录两幅图中降水量更大的数组
        if value2 > 2.5:
            max_array = max_array2
        elif value2 < 2.5 and value1 < value2:
            max_array = max_array2
        else:
            max_array = max_array1
    else:
        max_array = max_array1
    max_array.append(time)  # 添加时间信息
    print(max_array)

    grade = group_GPM(max_array[2])    # 调用group_GPM函数，将降水量最大值进行归类  1
    xy.append(grade)                   # 将归类属性添加到列表中去
    kk = xy
    # print(grade)
    # print(xy)

    result.append(time)     # 将时间信息写入数组
    result.append(kk)       # 将目标框信息写入数组

    max_point = max_array   # [行,列,最大值]
    max_level = grade

    # 输出行列号、最大值信息到指定数组
    if max_level == 0:
        datalist0.append(max_point)      # =0
    if max_level == 1:
        datalist1.append(max_point)      # <2.5
    if max_level == 2:
        datalist2.append(max_point)      # 2.5~16
    if max_level == 3:
        datalist3.append(max_point)     # > 16
    # print(result)
    matching.write(str(result) + "\n")
matching.close()

print(datalist2)
print(datalist3)


# 将变量写入表格
# 创建表格
book = xlwt.Workbook(encoding = "utf-8")
sheet1 = book.add_sheet("GPM_2.5",cell_overwrite_ok = True)
sheet2 = book.add_sheet("GPM_16",cell_overwrite_ok = True)
title = ["行","列","最大值"]

#将 属性元素写进表格
for i in range(3):
    sheet1.write(0, i, title[i])
    sheet2.write(0, i, title[i])

# GPM_2.5将数据写进表格
for i in range(len(datalist2)):
    data = datalist2[i]
    for j in range(3):
        sheet1.write(i+1,j,data[j])

# GPM_16将数据写进表格
for i in range(len(datalist3)):
    data = datalist3[i]
    for j in range(3):
        sheet2.write(i+1,j,data[j])

# 保存excel文件
savepath = "D:/GPM/2018_0601_points.xls"
book.save(savepath)

















