import os
# import netCDF4 as nc
from netCDF4 import Dataset
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import datetime
from osgeo import gdal, osr


# 读取h08影像的通道8,13,16，分别对应6.2微米、10.4微米、13.3微米
def read_h08(filepath): #读取华北地区的
    """
    :param filename: 影像名称
    :return: 三通道的数据，时间信息
    """
    rsname = os.path.split(filepath)[1]
    name = os.path.splitext(rsname)  # 区分文件名 和后缀，即('NC_H08_20180715_1030_R21_FLDK.06001_06001', '.nc')
    name1 = name[0].split("_")  # 'NC', 'H08', '20180715', '1030', 'R21', 'FLDK.06001', '06001']
    time = name1[2] + "_" + name1[3]

    # 使用netCDF4的Dataset方法即可读入文件，读入文件后直接输出即可查看文件的结构。
    fh = Dataset(filepath, mode='r')  # 把二进制文件转为可读的数字
    # print(fh)
    # 查看文件的变量
    # print(fh.variables.keys())
    c08 = fh.variables['tbb_08'][:]
    c13 = fh.variables['tbb_13'][:]
    c16 = fh.variables['tbb_16'][:]
    # # 经纬度
    # lon = fh.variables['longitude'][:]
    # lat = fh.variables['latitude'][:]

    # 插值成0.25度*0.25度，并根据经纬度进行裁剪（华北地区35-44N,110-122E），纬度左上角为（0,0）为44N ，经度左上角为110N
    c08_clip = cv2.resize(c08, (4801, 4801))[1340:1681, 980:1481]#[640:1001, 1200:1681] #(361,481)
    c13_clip = cv2.resize(c13, (4801, 4801))[1340:1681, 980:1481]#[640:1001, 1200:1681]
    c16_clip = cv2.resize(c16, (4801, 4801))[1340:1681, 980:1481]#[640:1001, 1200:1681]
    data_merge = cv2.merge([c08_clip, c13_clip, c16_clip])  # 合并通道
    return data_merge, time

# 读取h08影像的通道8,13,16，分别对应6.2微米、10.4微米、13.3微米
def read_h08_3c_allread(filepath):  #读取全部区域的 三通道影像
    """
    :param filename: 影像名称
    :return: 三通道的数据，时间信息
    """
    rsname = os.path.split(filepath)[1]
    name = os.path.splitext(rsname)  # 区分文件名 和后缀，即('NC_H08_20180715_1030_R21_FLDK.06001_06001', '.nc')
    name1 = name[0].split("_")  # 'NC', 'H08', '20180715', '1030', 'R21', 'FLDK.06001', '06001']
    time = name1[2] + "_" + name1[3]

    # 使用netCDF4的Dataset方法即可读入文件，读入文件后直接输出即可查看文件的结构。
    fh = Dataset(filepath, mode='r')  # 把二进制文件转为可读的数字
    # print(fh)
    # 查看文件的变量
    # print(fh.variables.keys())
    c08 = fh.variables['tbb_08'][:]
    c13 = fh.variables['tbb_13'][:]
    c16 = fh.variables['tbb_16'][:]
    # # 经纬度
    # lon = fh.variables['longitude'][:]
    # lat = fh.variables['latitude'][:]

    # 插值成0.25度*0.25度，并根据经纬度进行裁剪（华北地区35-44N,110-122E），纬度左上角为（0,0）为44N ，经度左上角为110N
    c08_clip = cv2.resize(c08, (4801, 4801))
    c13_clip = cv2.resize(c13, (4801, 4801))
    c16_clip = cv2.resize(c16, (4801, 4801))
    data_merge = cv2.merge([c08_clip, c13_clip, c16_clip])  # 合并通道
    return data_merge, time



# 读取h08影像的通道8,13,16，分别对应6.2微米、10.4微米、13.3微米
def read_h08_allchannel(filepath):
    """
    :param filename: 影像名称
    :return: 三通道的数据，时间信息
    """
    rsname = os.path.split(filepath)[1]
    name = os.path.splitext(rsname)  # 区分文件名 和后缀，即('NC_H08_20180715_1030_R21_FLDK.06001_06001', '.nc')
    name1 = name[0].split("_")  # 'NC', 'H08', '20180715', '1030', 'R21', 'FLDK.06001', '06001']
    time = name1[2] + "_" + name1[3]

    # 使用netCDF4的Dataset方法即可读入文件，读入文件后直接输出即可查看文件的结构。
    fh = Dataset(filepath, mode='r')  # 把二进制文件转为可读的数字
    # print(fh)
    # 查看文件的变量
    #print(fh.variables.keys())
    c01 = fh.variables["albedo_01"][:]
    c02 = fh.variables["albedo_02"][:]
    c03 = fh.variables["albedo_03"][:]
    c04 = fh.variables["albedo_04"][:]
    c05 = fh.variables["albedo_05"][:]
    c06 = fh.variables["albedo_06"][:]
    c07 = fh.variables['tbb_07'][:]
    c08 = fh.variables['tbb_08'][:]
    c09 = fh.variables['tbb_09'][:]
    c10 = fh.variables['tbb_10'][:]
    c11 = fh.variables['tbb_11'][:]
    c12 = fh.variables['tbb_12'][:]
    c13 = fh.variables['tbb_13'][:]
    c14 = fh.variables['tbb_14'][:]
    c15 = fh.variables['tbb_15'][:]
    c16 = fh.variables['tbb_16'][:]

    # # 经纬度
    # lon = fh.variables['longitude'][:]
    # lat = fh.variables['latitude'][:]

    # 插值成0.25度*0.25度，并根据经纬度进行裁剪（中国地区   35-44N,110-122E），纬度左上角为（0,0）为44N ，经度左上角为110N
    c01_inter = cv2.resize(c01, (4801, 4801))[0:2401, 0:1801]
    c02_inter = cv2.resize(c02, (4801, 4801))[0:2401, 0:1801]
    c03_inter = cv2.resize(c03, (4801, 4801))[0:2401, 0:1801]
    c04_inter = cv2.resize(c04, (4801, 4801))[0:2401, 0:1801]
    c05_inter = cv2.resize(c05, (4801, 4801))[0:2401, 0:1801]
    c06_inter = cv2.resize(c06, (4801, 4801))[0:2401, 0:1801]
    c07_inter = cv2.resize(c07, (4801, 4801))[0:2401, 0:1801]
    c08_inter = cv2.resize(c08, (4801, 4801))[0:2401, 0:1801]
    c09_inter = cv2.resize(c09, (4801, 4801))[0:2401, 0:1801]
    c10_inter = cv2.resize(c10, (4801, 4801))[0:2401, 0:1801]
    c11_inter = cv2.resize(c11, (4801, 4801))[0:2401, 0:1801]
    c12_inter = cv2.resize(c12, (4801, 4801))[0:2401, 0:1801]
    c13_inter = cv2.resize(c13, (4801, 4801))[0:2401, 0:1801]
    c14_inter = cv2.resize(c14, (4801, 4801))[0:2401, 0:1801]
    c15_inter = cv2.resize(c15, (4801, 4801))[0:2401, 0:1801]
    c16_inter = cv2.resize(c16, (4801, 4801))[0:2401, 0:1801]
    data_merge = cv2.merge([c01_inter, c02_inter, c03_inter, c04_inter, c05_inter, c06_inter, c07_inter, c08_inter, c09_inter, c10_inter, c11_inter,
                            c12_inter, c13_inter, c14_inter, c15_inter, c16_inter])  # 合并通道
    return data_merge, time



def read_tif(filepath):
    """
    :param filepath: 华南地区的影像，三个通道，格式为tif
    :return:
    """
    rsname = os.path.split(filepath)[1]
    name = os.path.splitext(rsname)  # 区分文件名 和后缀，即('NC_H08_20180715_1030_R21_FLDK.06001_06001', '.nc')
    name1 = name[0].split("_")  # 'NC', 'H08', '20180715', '1030', 'R21', 'FLDK.06001', '06001']
    time = name1[2] + "_" + name1[3]
    data = cv2.imread(filepath, -1)
    return data, time  #(341,501,3)

"""
（3）目标框提取并保存
"""
# 该函数是用来统计面积（函数参数输入为数组）
# 原理：针对目标框进行计算，目标框内像素为0则为背景，不计入。
def Calculated_area(data):
    count = 0  # 近红外通道像素有值的数量
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 0:  # 针对目标框进行计算，目标框内像素为0则为背景，不计入
                pass
            else:
                count = count + 1
    return count


# 统计下目标框中心，输入imgpath为三通道数据，输出 (x1, y1, w1, h1, centerx, centery)
def cal_retangle_center(imgdata, areamin=1, areamax=96):
    allimg = np.float32(imgdata)  # 3通道数据
    allimg2 = imgdata.copy()  # 重新复制一个图像

    gray = cv2.cvtColor(allimg, cv2.COLOR_BGR2GRAY)  # 变为灰度图
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)  # 阈值分割得到二值化图片(阈值、二值化图像)，参数cv2.THRESH_BINARY是

    binary = np.uint8(binary)  # 一定是此类型，否则后续会出错
    imgg, contours, heriachy = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)  # 自动提取轮廓， contours是轮廓的坐标信息，列表形式； heriachy图像
    """
    自动绘制多个轮廓，及多个轮廓的矩形边界
    """
    retangle_all = []  # 矩形边框(编号，左下角x，左下角y，宽，长)
    for i, contour in enumerate(contours):
        # print(i, contour)
        cv2.drawContours(allimg, contours, i, (0, 255, 0),
                         1)  # allimg必须是三通道的图像，才能显示轮廓,#第一个参数指在哪幅图上绘制轮廓信息，第二个参数是轮廓本身，第三个参数是指定绘制哪条轮廓
        # 第四个参数是绘图的颜色，第五个参数是绘制的线宽 输入-1则表示填充

        x, y, w, h = cv2.boundingRect(contours[i])  # 外接框
        #print(x, y, w, h)  # x,y 为左上角坐标
        img_rectangle = cv2.rectangle(allimg, (x, y), (x + w, y + h), (255, 0, 0), 1)
        # 矩形边框添加文字（编号）
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_rectangle, str(i), (x + w, y + h), font, 0.30, (0, 255, 0), 1)
        # 汇总所有矩形边框的位置信息（左上角，宽，长）
        retangle_all.append((i, x, y, w, h))

    print("初次提取的轮廓retangle_all", retangle_all)

    # cv2.namedWindow("detect contours", 0) #名字保持一致，可放大窗口
    # cv2.imshow("detect contours", allimg)
    # cv2.imwrite("E:/H08/GPM-V2/obj/t2.tif", allimg)
    # cv2.waitKey(0)

    """
    批量统计目标的面积，设置阈值
    """
    roi_area_threshold = []  # 满足阈值条件的目标框（编号，面积数）
    for i in range(len(retangle_all)):
        x, y, w, h = retangle_all[i][1], retangle_all[i][2], retangle_all[i][3], retangle_all[i][4]
        # img = cv2.split(allimg)[1] #根据13通道中的数值计算面积
        img = allimg[:, :, 1]  # 根据13通道中的数值计算面积
        roi = img[y:y + h, x:x + w]
        roi_area = Calculated_area(np.array(roi))
        # print("编号：", i, "面积：", roi_area)
        if roi_area > areamin and roi_area < areamax:  # 面积阈值
            roi_area_threshold.append((i, roi_area))
        else:
            pass
    print("符合面积阈值条件的目标框（编号,面积）: ", roi_area_threshold)

    """
    重新在原图上显示出满足条件的目标框，并计算目标框的中心
    批量处理
    """
    center = []  # 统计所有满足条件的目标框的中心位置,(x1, y1, w1, h1, centerx, centery)
    for m in range(len(roi_area_threshold)):
        roi = roi_area_threshold[m][0]  # 获取满足阈值条件的编号
        roi2 = retangle_all[roi]  # 获取满足阈值条件的编号的 目标框坐标信息（编号，x,y,w,h）
        x1, y1, w1, h1 = roi2[1], roi2[2], roi2[3], roi2[4]
        img_rectangle_th = cv2.rectangle(allimg2, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 1)  # 截取出感兴趣的区域
        # 计算矩形中心
        centerx = int((x1 + x1 + w1) / 2)
        centery = int((y1 + y1 + h1) / 2)
        center.append([x1, y1, w1, h1, centerx, centery])
        # cv2.imwrite("retangle_th_2.tif", img_rectangle_th)
    return center

# imgdata = yz[:, :, :, 1]  # 第一时刻的H08经过阈值处理后的三通道数据
# cal_retangle_center(imgdata)
# print(cal_retangle_center(imgdata))

"""
(3）按照最小距离匹配前后两个时刻的云

以第二时刻为标准计算距离，能克服云消失的情况，但是由于有云会突然冒出，所以第二时刻目标框会出现一对多的情况，因此需要进一步删选目标框

"""

##根据索引值，调取相对应的矩形坐标信息，并截取目标框的范围
# 根据目标框，截取潜在的对流云
# 初定对流云的坐标信息（x,y,w,h,centerx,centery）
def final_could(imgdata, retangle):  # imgdata为单通道数据, retangle是目标框的坐标信息
    cloud1_t1_retangle = retangle[0:4]  # x,y,w,h
    x = cloud1_t1_retangle[0]
    y = cloud1_t1_retangle[1]
    w = cloud1_t1_retangle[2]
    h = cloud1_t1_retangle[3]
    final_cloud = imgdata[y:y + h, x:x + w]  # 最后的目标框内的单通道数据(h,w )即是高，宽，即是列和行       就是图片切割的w,h是宽和高，而数组讲的是行（row）和列（column）
    # cv2.imwrite("E:/H08/CI2/clip/roi/"+savename+".tif", final_cloud)
    return final_cloud, retangle

# 【取目标框中10%最暖像素】将每个通道的像素值从高到低排序，然后取其10%，求平均，将其作为云团的像素值（采取方式：将其转成list ,去除0值，然后排序，截取）
def sort_mean_bs(data):
    f2 = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 0:
                pass
            else:
                f2.append(data[i, j])
    f2_sort = sorted(f2)
    count = len(f2_sort)
    count = math.ceil(count * 0.1)  # 返回大于或等于一个给定数字的最小整数。
    f2_sort_clip = f2_sort[0: count]
    f2_sort_clip_average = np.mean(f2_sort_clip)  # 求取平均亮温,单位是K
    return f2_sort_clip_average

#---------------------------------------------------------------------------------------------------------------------------

def ci(cal_t1_retangle_center, cal_t2_retangle_center):  # cal_t1_retangle_center t1时刻的目标框

    # 计算点点之间的距离
    all_distance = []  # i, j, d  第二时刻的编号，第一时刻的编号，距离值
    for i in range(len(cal_t2_retangle_center)):
        for j in range(len(cal_t1_retangle_center)):
            x1 = cal_t1_retangle_center[j][4]  # 行
            y1 = cal_t1_retangle_center[j][5]  # 列
            x2 = cal_t2_retangle_center[i][4]  # 行
            y2 = cal_t2_retangle_center[i][5]  # 列
            d = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)  # 计算距离
            all_distance.append([i, j, d])
    print("all_distance:", len(all_distance), all_distance)

    obj_distance = []  # 提取出距离值
    count1 = len(cal_t1_retangle_center)  # 第一时刻的目标框个数
    count2 = len(cal_t2_retangle_center)  # 第二时刻的目标框个数
    print("第一时刻的个数", count1)
    print("第二时刻的个数", count2)



    for i in range(len(all_distance)):
        obj_distance.append(all_distance[i][2])
    print("全部的距离值：", len(obj_distance), obj_distance)

    d_min = []  # 计算前后时刻的距离最小值,len(d_min)  应该和第二时刻的个数相同
    t1_id_all = []  # 第一时刻对应的索引
    t2_id_all = []  # 第二时刻对应的索引

    all_cloud_t1 = []  # 第一时刻的所有目标框的位置信息，len(all_cloud_t1)表示目标框的个数
    all_cloud_t2 = []  # 第二时刻的所有目标框的位置信息
    for m in range(count2):  # 要按照第二时刻进行计算距离
        dis_min = np.min(obj_distance[count1 * m:count1 * (m + 1)])  # 计算区间内的最小值
        if dis_min < 15:  # 两点之间的距离要是小于15，则保留，可再次判别前后两个时刻为同一朵云
            d_min.append(dis_min)

            dis_index = obj_distance[count1 * m:count1 * (m + 1)].index(
                dis_min)  # 某个区间（obj_distance[count1 * m:count1 * (m + 1)]）内的值，也就是第一时刻对应的索引
            all_cloud_t1.append(cal_t1_retangle_center[dis_index])
            t1_id_all.append(dis_index)

            t2_id1 = all_distance[count1 * m:count1 * (m + 1)][dis_index][0]  # 提取出第二时刻对应的索引
            all_cloud_t2.append(cal_t2_retangle_center[t2_id1])
            t2_id_all.append(t2_id1)
        else:
            pass
    print("全部的距离最小值", d_min)
    print("全部的距离最小值的个数", len(d_min))
    print("第一时刻对应的索引：", len(t1_id_all), t1_id_all)
    print("第二时刻对应的索引：", len(t2_id_all), t2_id_all)

    print("第一时刻目标框all_cloud_t1", len(all_cloud_t1), all_cloud_t1)
    print("第二时刻目标框all_cloud_t2", len(all_cloud_t2), all_cloud_t2)

    # 上面以第二时刻为标准计算距离，能克服云消失的情况，但是由于有云会突然冒出，所以会出现在第二时刻目标框出现一对多的情况，因此需要进一步删选目标框
    # 针对前一时刻的编码号，不能有重复，若是有重复说明，说明有问题

    all_cloud_t1_save = []
    all_cloud_t2_save = []
    myset = set(t1_id_all)
    print("t1时刻编号：", myset)
    for item in myset:
        if t1_id_all.count(item) > 1:
            # 重复编号的下标索引值
            Duplicate_index = [i for i, val in enumerate(t1_id_all) if val == item]
            print("重复编号的下标索引值:", Duplicate_index)  # [0,1,5]  即表示在第一和第二个位置,第5个位置
            # #remove_id.append((item, Duplicate_index)) #[0,(0,1)]
            d_infor = []  # 例如[(0, 17.49), (1, 2.82), (5, 10)]
            for i in range(len(Duplicate_index)):
                d_infor.append((Duplicate_index[i], d_min[Duplicate_index[i]]))  # 保存重复编号的下标，面积
            print("d_infor", d_infor)
            t1_id = [d_infor[i][0] for i in
                     range(len(d_infor))]  # T1的重复编号的下标索引  [0, 1, 5]  即表示 编号0 的重复位置在第一个  第二个  第六个
            d_dis = [d_infor[i][1] for i in range(len(d_infor))]  # 距离 [ 17.49, 2.82, 10]

            da = sorted(d_dis)  # 从小到大排序 [2.82, 10, 17.49]
            print("da:", da)

            da_save = da[0]  # 提取最小的距离
            di = d_dis.index(da_save)
            t1_id_one = t1_id[di]
            aa = all_cloud_t2[t1_id_one]
            all_cloud_t2_save.append(aa)

            bb = all_cloud_t1[t1_id_one]
            all_cloud_t1_save.append(bb)

        else:
            single_id = t1_id_all.index(item)
            # print("保留的编码号：", single_id)
            all_cloud_t1_save.append(all_cloud_t1[single_id])
            all_cloud_t2_save.append(all_cloud_t2[single_id])

    print("all_cloud_t1_save :", len(all_cloud_t1_save), all_cloud_t1_save)
    print("all_cloud_t2_save :", len(all_cloud_t2_save), all_cloud_t1_save)

    all_cloud_t1 = all_cloud_t1_save
    all_cloud_t2 = all_cloud_t2_save
    print("第一时刻更新后的目标框all_cloud_t1", len(all_cloud_t1), all_cloud_t1)
    print("第二时刻更新后的目标框all_cloud_t2", len(all_cloud_t2), all_cloud_t2)

    return all_cloud_t1, all_cloud_t2  # 返回亮温比率，第一时刻及第二时刻的符合条件的目标框信息



def updata_obj(cal_t1_retangle_center, cal_t2_retangle_center): # cal_t1_retangle_center t1时刻的目标框

    # 计算点点之间的距离
    all_distance = []  # i, j, d  第二时刻的编号，第一时刻的编号，距离值
    for i in range(len(cal_t2_retangle_center)):
        for j in range(len(cal_t1_retangle_center)):
            x1 = cal_t1_retangle_center[j][4]  # 行
            y1 = cal_t1_retangle_center[j][5]  # 列
            x2 = cal_t2_retangle_center[i][4]  # 行
            y2 = cal_t2_retangle_center[i][5]  # 列
            d = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) #计算距离
            all_distance.append([i, j, d])
    print("all_distance:", len(all_distance), all_distance)

    obj_distance = []  # 提取出距离值
    count1 = len(cal_t1_retangle_center)  # 第一时刻的目标框个数
    count2 = len(cal_t2_retangle_center)  # 第二时刻的目标框个数
    print("第一时刻的个数", count1)
    print("第二时刻的个数", count2)


    for i in range(len(all_distance)):
        obj_distance.append(all_distance[i][2])
    print("全部的距离值：", len(obj_distance), obj_distance)

    d_min = []  # 计算前后时刻的距离最小值,len(d_min)  应该和第二时刻的个数相同
    t1_id_all = []  # 第一时刻对应的索引
    t2_id_all = []  #第二时刻对应的索引

    all_cloud_t1 = []  # 第一时刻的所有目标框的位置信息，len(all_cloud_t1)表示目标框的个数
    all_cloud_t2 = []  #第二时刻的所有目标框的位置信息
    for m in range(count2):  # 要按照第二时刻进行计算距离
        if m==0:
            pass
        else:
            dis_min = np.min(obj_distance[count1 * m:count1 * (m + 1)])  #计算区间内的最小值
            if dis_min < 15:  #两点之间的距离要是小于15，则保留，可再次判别前后两个时刻为同一朵云
                d_min.append(dis_min)

                dis_index = obj_distance[count1 * m:count1 * (m + 1)].index(dis_min)#  某个区间（obj_distance[count1 * m:count1 * (m + 1)]）内的值，也就是第一时刻对应的索引
                all_cloud_t1.append(cal_t1_retangle_center[dis_index])
                t1_id_all.append(dis_index)

                t2_id1 = all_distance[count1 * m:count1 * (m + 1)][dis_index][0] # 提取出第二时刻对应的索引
                all_cloud_t2.append(cal_t2_retangle_center[t2_id1])
                t2_id_all.append(t2_id1)
            else:
                pass
    print("全部的距离最小值", d_min)
    print("全部的距离最小值的个数", len(d_min))
    print("第一时刻对应的索引：", len(t1_id_all), t1_id_all)
    print("第二时刻对应的索引：", len(t2_id_all), t2_id_all)

    print("第一时刻目标框all_cloud_t1", len(all_cloud_t1), all_cloud_t1)
    print("第二时刻目标框all_cloud_t2", len(all_cloud_t2), all_cloud_t2)



# 上面以第二时刻为标准计算距离，能克服云消失的情况，但是由于有云会突然冒出，所以会出现在第二时刻目标框出现一对多的情况，因此需要进一步删选目标框
# 针对前一时刻的编码号，不能有重复，若是有重复说明，说明有问题

    all_cloud_t1_save = []
    all_cloud_t2_save = []
    myset = set(t1_id_all)
    print("t1时刻编号：", myset)
    for item in myset:
        if t1_id_all.count(item) > 1:
            # 重复编号的下标索引值
            Duplicate_index = [i for i, val in enumerate(t1_id_all) if val == item]
            print("重复编号的下标索引值:", Duplicate_index)  # [0,1,5]  即表示在第一和第二个位置,第5个位置
            # #remove_id.append((item, Duplicate_index)) #[0,(0,1)]
            d_infor = []  # 例如[(0, 17.49), (1, 2.82), (5, 10)]
            for i in range(len(Duplicate_index)):
                d_infor.append((Duplicate_index[i], d_min[Duplicate_index[i]]))  # 保存重复编号的下标，面积
            print("d_infor", d_infor)
            t1_id = [d_infor[i][0] for i in range(len(d_infor))]  # T1的重复编号的下标索引  [0, 1, 5]  即表示 编号0 的重复位置在第一个  第二个  第六个
            d_dis = [d_infor[i][1] for i in range(len(d_infor))]  # 距离 [ 17.49, 2.82, 10]

            da = sorted(d_dis)  # 从小到大排序 [2.82, 10, 17.49]
            print("da:", da)

            da_save = da[0] #提取最小的距离
            di = d_dis.index(da_save)
            t1_id_one = t1_id[di]
            aa = all_cloud_t2[t1_id_one]
            all_cloud_t2_save.append(aa)

            bb = all_cloud_t1[t1_id_one]
            all_cloud_t1_save.append(bb)

        else:
            single_id = t1_id_all.index(item)
            # print("保留的编码号：", single_id)
            all_cloud_t1_save.append(all_cloud_t1[single_id])
            all_cloud_t2_save.append(all_cloud_t2[single_id])

    print("all_cloud_t1_save :", len(all_cloud_t1_save), all_cloud_t1_save)
    print("all_cloud_t2_save :", len(all_cloud_t2_save), all_cloud_t1_save)

    all_cloud_t1 = all_cloud_t1_save
    all_cloud_t2 = all_cloud_t2_save
    print("第一时刻更新后的目标框all_cloud_t1", len(all_cloud_t1), all_cloud_t1)
    print("第二时刻更新后的目标框all_cloud_t2", len(all_cloud_t2), all_cloud_t2)

    return all_cloud_t1, all_cloud_t2   # 返回亮温比率，第一时刻及第二时刻的符合条件的目标框信息

#-----------------------下面函数用于H08数据预处理--------------------------------
"""
提取阈值         
"""
def yz(data_3c,w,h):
    yz = np.zeros((w, h, 3))  # 后续可以替换掉
    for m in range(data_3c.shape[0]):
        for n in range(data_3c.shape[1]):
            if 0 < data_3c[m, n, 1] < 273:
                yz[m, n, :] = data_3c[m, n, :]
            else:
                yz[m, n, :] = 0
    return yz


"""
将背景值0变成nan，便于后续计算最小值
使用np.nanmin()
"""
def b_nan(data): #data为二维数组
    newdata = np.ones_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 0:
                newdata[i, j] = "nan"
            else:
                newdata[i, j] = data[i, j]
    return newdata


def data_min_index(t1_could, all_cloud_t1):  #目标区域，目标框信息   针对的是一个框的
    """
    求目标框中 c13通道最小值所在的位置索引（即行列号）

    :param t1_could: 目标区域的c13通道
    :param all_cloud_t1: 单个目标框的位置信息（x,w,w,h,cx,cy）
    :return: c13通道最小值所在的位置索引（即行列号）
    """
    #把背景值0 变成nan
    t1_could_nan = b_nan(t1_could)
    t1_row_col = np.unravel_index(np.nanargmin(t1_could_nan), t1_could_nan.shape) #np.nanargmin() 求取除nan值以外的 最小值，，返回是 行列索引

    row = t1_row_col[0]
    col = t1_row_col[1]

    true_row = all_cloud_t1[1]+row
    true_col = all_cloud_t1[0]+col
    #print(true_col, true_row)  #列 行  对于 x y
    return true_col, true_row

"""
根据缺失时刻的前后两个时刻，c13通道像素最小值移动的距离，计算风速；
前一时刻根据风速，整体进行移动（缺点是形状大小不变）

定义个移动函数

"""
def displacement(data, sx, sy, time=None):
    """
    :param data: 缺失时刻的前或后时刻的影像（三个通道）,(w,h,c)
    :param sx:  列方向风速移动的格点数
    :param sy:  行方向风速移动的格点数
    :param time: 指定前后时刻,t1是前一时刻，t2是后一时刻
    :return: 缺失时刻的影像（三通道）
    """
    data_new = np.zeros_like(data)
    for i in range(abs(round(sy / 2)),  data.shape[0] - abs(round(sy / 2))):  # 这是行
        for j in range(abs(round(sx / 2)), data.shape[1] - abs(round(sx / 2))):  # 这是列
            if time == "t1":
                data_new[i + round(sy / 2), j + round(sx / 2), :] = data[i, j, :]
            elif time == "t2":
                data_new[i - round(sy / 2), j - round(sx / 2), :] = data[i, j, :]
    return data_new

#保存浮点数的数据
def write_img(filename, im_data):
    """

    :param filename: 保存的影像的路径及名称 如C:/1.TIF
    :param im_data: 数据，(channel,w,h)
    :return:
    """
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



