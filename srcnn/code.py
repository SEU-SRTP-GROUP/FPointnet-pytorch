import numpy as np
import io
from PIL import Image, ImageTk
import tkinter as tk

import cv2
import math

def nearest_neighbor(input, zoom_multiples):
    '''
    最近邻插值（适用于灰度图）
    :param input_signal: 输入图像
    :param zoom_multiples:  缩放倍数
    :return: 缩放后的图像
    '''
    input_signal_cp = np.copy(input)   # 输入图像的副本

    input_row, input_col = input_signal_cp.shape # 输入图像的尺寸（行、列）

    # 输出图像的尺寸
    output_row = int(input_row * zoom_multiples)
    output_col = int(input_col * zoom_multiples)

    output_signal = np.zeros((output_row, output_col)) # 输出图片

    for i in range(output_row):
        for j in range(output_col):
            # 输出图片中坐标 （i，j）对应至输入图片中的（m，n）
            m = round(i / output_row * input_row)
            n = round(j / output_col * input_col)
            # 防止四舍五入后越界
            if m >= input_row:
                m = input_row - 1
            if n >= input_col:
                n = input_col - 1
            # 插值
            output_signal[i, j] = input_signal_cp[m, n]

    return output_signal

def double_linear(input_signal, zoom_multiples):
    '''
    双线性插值
    :param input_signal: 输入图像
    :param zoom_multiples: 放大倍数
    :return: 双线性插值后的图像
    '''
    input_signal_cp = np.copy(input_signal)   # 输入图像的副本

    input_row, input_col = input_signal_cp.shape # 输入图像的尺寸（行、列）

    # 输出图像的尺寸
    output_row = int(input_row * zoom_multiples)
    output_col = int(input_col * zoom_multiples)

    output_signal = np.zeros((output_row, output_col)) # 输出图片

    for i in range(output_row):
        for j in range(output_col):
            # 输出图片中坐标 （i，j）对应至输入图片中的最近的四个点点（x1，y1）（x2, y2），（x3， y3），(x4，y4)的均值
            temp_x = i / output_row * input_row
            temp_y = j / output_col * input_col

            x1 = int(temp_x)
            y1 = int(temp_y)

            x2 = x1
            y2 = y1 + 1

            x3 = x1 + 1
            y3 = y1

            x4 = x1 + 1
            y4 = y1 + 1

            u = temp_x - x1
            v = temp_y - y1

            # 防止越界
            if x4 >= input_row:
                x4 = input_row - 1
                x2 = x4
                x1 = x4 - 1
                x3 = x4 - 1
            if y4 >= input_col:
                y4 = input_col - 1
                y3 = y4
                y1 = y4 - 1
                y2 = y4 - 1

            # 插值
            output_signal[i, j] = (1-u)*(1-v)*int(input_signal_cp[x1, y1]) + (1-u)*v*int(input_signal_cp[x2, y2]) + u*(1-v)*int(input_signal_cp[x3, y3]) + u*v*int(input_signal_cp[x4, y4])
    return output_signal
def resize(w, h, w_box, h_box, pil_image):
    f1 = 1.0 * w_box / w  # 1.0 forces float division in Python2
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)

if __name__ =='__main__':
    img1 = cv2.imread("1.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    result = nearest_neighbor(img1,10)
    cv2.imwrite("final.jpg", result)



    root = tk.Tk()
    # 期望图像显示的大小
    w_box = 800
    h_box = 1000

    # 以一个PIL图像对象打开
    pil_image = Image.open('final.jpg')

    # 获取图像的原始大小
    w, h = pil_image.size

    # but fits into the specified display box
    # 缩放图像让它保持比例，同时限制在一个矩形框范围内
    pil_image_resized = resize(w, h, w_box, h_box, pil_image)

    # 把PIL图像对象转变为Tkinter的PhotoImage对象
    tk_image = ImageTk.PhotoImage(pil_image_resized)

    # Label: 这个小工具，就是个显示框，小窗口，把图像大小显示到指定的显示框
    label = tk.Label(root, image=tk_image, width=w_box, height=h_box)
    # padx,pady是图像与窗口边缘的距离
    label.pack(padx=5, pady=5)
    root.mainloop()
