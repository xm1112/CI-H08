
"""
2022/9/14
xm
遍历文件，数据，存储为（W,H,C,T）
提取阈值，和面积大小
跟踪（前后两个时刻的计算），使用最小距离；并以第二时刻为基准，能克服云的突然冒出情况；并进行一对一删选；并根据风速设置最大距离，克服误检情况。

批量处理--处理一个月的文件夹

文件结构：
2018/ 20180601/ 0000.nc  0010.nc ...
      20180602/ 0000.nc  0010.nc ...
      20180603/ 0000.nc  0010.nc ...
      ......

V4版本比V3版本实现目的是一样的，只是读取数据的存入方式不一样
v4是直接从文件中读取.nc格式
v3是从文件中读取.nc格式，并存储为（w,h,c,time）

"""
import os
import netCDF4 as nc
from netCDF4 import Dataset
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import datetime


# 读取h08影像的通道8,13,16，分别对应6.2微米、10.4微米、13.3微米
def read_h08(filepath):
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

def ci(t1_imgdata, t2_imgdata, cal_t1_retangle_center, cal_t2_retangle_center, path, name=None): # imgdata三通道数据，cal_t1_retangle_center t1时刻的目标框

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
        if m == 0:
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

    # 克服云突然出现后，在图像上显示目标框
    allimg3 = t2_imgdata.copy()# imgdata三通道数据
    for i in range(len(all_cloud_t2)):
        x1 = all_cloud_t2[i][0]
        y1 = all_cloud_t2[i][1]
        w1 = all_cloud_t2[i][2]
        h1 = all_cloud_t2[i][3]
        img_end_could = cv2.rectangle(allimg3, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 1)
        # tru = cv2.rectangle(allimg3, (0, 0), (20, 20), (255, 0, 0), 1)
        cv2.imwrite(path+str(name)+".tif", img_end_could)

    #开始遍历所有时序和所有目标框，寻找满足条件的目标框，并保存经纬度信息及时间信息
    all_data = []
    diff = []  #汇总所有亮温差比率

    for j in range(len(all_cloud_t2)):
        t1_could = final_could(t1_imgdata[:, :, 1], all_cloud_t1[j])[0]  # t1_imgdata[:, :, 1]表示t1_c13通道， all_cloud_t1[j]表示为第一时刻的目标框第j个
        t2_could = final_could(t2_imgdata[:, :, 1], all_cloud_t2[j])[0]
        t1_cttr = sort_mean_bs(t1_could)  # t1_could 是目标框1的区域，，需要用10.4微米的波段，即取其10%暖像素平均
        t2_cttr = sort_mean_bs(t2_could)
        # 前后两时刻的亮温差比率
        diff_t2_t1 = (t2_cttr - t1_cttr) * 6#每小时
        diff.append(diff_t2_t1)
        if diff_t2_t1 <= -16:  #大于16
            all_data.append((diff_t2_t1, all_cloud_t2[j], all_cloud_t1[j])) #保留前一时刻的目标框  #全部放在这
        else:
            pass
    # print("再次更新后的第二时刻的目标框信息：", all_data)
    return all_data   #返回亮温比率，第一时刻及第二时刻的符合条件的目标框信息

def yz(data_3c):
    # w = data_3c.shape[0]
    # h = data_3c.shape[1]
    yz = np.zeros((341, 501, 3))  # 后续可以替换掉
    for m in range(data_3c.shape[0]):
        for n in range(data_3c.shape[1]):
            if 0 < data_3c[m, n, 1] < 273:
                yz[m, n, :] = data_3c[m, n, :]
            else:
                yz[m, n, :] = 0
    return yz

#_________________________________________________________________________________________
if __name__ == '__main__':

    starttime = datetime.datetime.now()
    fpath = "H:/Himawari-8/Chinal_S/2018/"
    path = sorted(os.listdir(fpath))
    print("一级文件：", path)
    print("共读取%d天影像" % len(path))

    file = [] #包含所有影像的具体路径
    all_count = 0 #计算所有影像数据
    for i in range(77, 78):#len(path)0,61为 6-7月31
        imgpath = os.path.join(fpath, path[i])  # "H:/Himawari-8/2018/20180601/"
        fname = path[i]
        print('年_月_日：', fname)  # 20180601
        #os.makedirs("E:/H08/GPM-V2/"+fname)
        ten_img = sorted(os.listdir(imgpath))  # 142张影像,一定要排序
        print(ten_img)
        all_count = all_count + len(ten_img)
        for j in range(len(ten_img)):  # len(img)
            datapath = os.path.join(imgpath, ten_img[j])
            file.append(datapath)
            print("影像所在路径：", datapath)  # H:/Himawari-8/2018/20180601/NC_H08_20180602_1330_R21_FLDK.06001_06001.nc

    file = sorted(file)  #进行排序
    print("-------sorted(file):", file)

    time_count = all_count  # len(path)  一天的影像个数
    print("共输入%d张影像" % time_count)

    # alldata = np.ones((361, 481, 3, time_count))  # 存储时序数据
    time_information = []  # 存储时间信息

    for i in range(time_count):
        time = read_h08(os.path.join(file[i]))[1]
        time_information.append(time)

    path = "E:/H08/GPM-V2/20180817-v4/"  #保存 目标框图片

    # 开始遍历时序数据
    ci_all = []
    for i in range(29, 35): #(0,time_count-1)
        img1 = read_h08(os.path.join(file[i]))[0]
        img2 = read_h08(os.path.join(file[i+1]))[0]
        print("数据名字：", read_h08(os.path.join(file[i+1]))[1])

        t1_imgdata = yz(img1)
        t2_imgdata = yz(img2)
        cal_t1_retangle_center = cal_retangle_center(t1_imgdata)  # t1时刻的三通道数据   # ([x, y, w, h, centerx, centery])),输入三通道
        cal_t2_retangle_center = cal_retangle_center(t2_imgdata)  # t2时刻的三通道数据

        print("----------------矩形坐标信息----------------------")
        print("第一时刻的目标框坐标信息：", cal_t1_retangle_center)
        print("第二时刻的目标框坐标信息：", cal_t2_retangle_center)

        ci1 = ci(t1_imgdata, t2_imgdata, cal_t1_retangle_center, cal_t2_retangle_center, path, name=time_information[i+1])
        ci_all.append((time_information[i+1], ci1)) #第二张影像算起
        print("%d时刻写入txt中" % i)

    #保留每个时刻的亮温差比率大于16的目标框
    file = open('2018-0703-y16.txt', 'w', encoding='utf-8')
    for j in range(len(ci_all)):  #j表示第几张
        file.write(str(ci_all[j]) + '\n')
    file.close()

    endtime = datetime.datetime.now()
    print("用时：", endtime - starttime)
    # print((endtime - starttime).seconds) #秒
