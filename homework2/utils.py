import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from struct import unpack

def byte_to_int(str1):
    #从一个str类型的byte到int
    result = 0
    for i in range(len(str1)):
        y = int(str1[len(str1) - 1 -i])
        result += y*2**i
    return result


def breakup_byte(num1, n):
    #byte为输入的类型为byte的参数，n为每个数要的位数
    result = []
    num = num1[2:]



def bmp2hist(filename):
    xxx=1
    #列出1, 4, 8, 16, 24图的位置
    imgs = os.listdir(filename)
    imgs_path = []
    print(imgs)

    for img_name in imgs:
        img_path = filename + os.sep + img_name
        imgs_path.append(img_path)

    #执行
    for img_path in imgs_path:
        img = open(img_path, "rb")
        #跳过bmp文件信息的开头，直接读取图片的size信息
        #字节数
        img.seek(28)
        bit_num = unpack("<i", img.read(4))[0]

        #数据偏移
        img.seek(10)
        to_img_data = unpack("<i", img.read(4))[0]

        #width height
        img.seek(4, 1)
        img_width = unpack("<i", img.read(4))[0]
        img_height = unpack("<i", img.read(4))[0]

        #颜色索引数
        img.seek(50)
        color_num = unpack("<i", img.read(4))[0]

        #1位每个像素一位，4位一个像素0.5字节，8位一个像素1字节，16位一个像素2字节
        #读取指针总共跳过54位到颜色盘，其中16，24位图像不需要调色盘
        img.seek(54)
        if(bit_num <= 8):
            #多少字节调色板颜色就有2^n个
            color_table_num = 2**int(bit_num)
            color_table = np.zeros((color_table_num, 3), dtype=np.int)
            for i in range(color_table_num):
                b = unpack("B", img.read(1))[0];
                g = unpack("B", img.read(1))[0];
                r = unpack("B", img.read(1))[0];
                alpha = unpack("B", img.read(1))[0];
                color_table[i][0] = b;
                color_table[i][1] = g;
                color_table[i][2] = r;
            
        #将数据存入numpy中
        img.seek(to_img_data)
        img_np = np.zeros((img_height, img_width, 3), dtype=np.int)
        
        #计算读入的总字节数
        num = 0
        #数据排列从左到右，从上到下
        x = 0
        y = 0
        while y < img_height:
            while x < img_width:
                if (bit_num <= 8):
                    img_byte = unpack("B", img.read(1))[0]
                    img_byte = bin(img_byte)
                    color_index = breakup_byte(img_byte, bit_num)





bmp2hist("./data-ORL/ORL/")