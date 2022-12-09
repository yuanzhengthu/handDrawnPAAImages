'''
created on 2022-08-25
author: yuanthu
description: painting blood vessel
'''

from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox, QPoint, qRed, qBlue, qGreen, QPainter, QImage, QRect
from PyQt5 import QtGui
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QSplitter, QComboBox, QLabel, QSpinBox, QFileDialog
from PaintBoard import PaintBoard
import cv2
import numpy as np


# 推荐方式
def QImage2CV(qimg):
    tmp = qimg

    # 使用numpy创建空的图象
    cv_image = np.zeros((tmp.height(), tmp.width(), 3), dtype=np.uint8)

    for row in range(0, tmp.height()):
        for col in range(0, tmp.width()):
            r = qRed(tmp.pixel(col, row))
            g = qGreen(tmp.pixel(col, row))
            b = qBlue(tmp.pixel(col, row))
            cv_image[row, col, 0] = b
            cv_image[row, col, 1] = g
            cv_image[row, col, 2] = r

    return cv_image


def get_noise(img, value=10):
    '''
    #生成噪声图像 >>> 输入： img图像

        value= 大小控制雨滴的多少 >>> 返回图像大小的模糊噪声图像
    '''

    noise = np.random.uniform(0, 256, img.shape[0:2])
    # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    # 噪声做初次模糊
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)

    # 可以输出噪声看看
    '''cv2.imshow('img',noise)
    cv2.waitKey()
    cv2.destroyWindow('img')'''
    return noise


def rain_blur(noise, length=10, angle=0, w=1):
    '''
    将噪声加上运动模糊,模仿雨滴  >>>输入
    noise：输入噪声图，shape = img.shape[0:2]
    length: 对角矩阵大小，表示雨滴的长度
    angle： 倾斜的角度，逆时针为正
    w:      雨滴大小  >>>输出带模糊的噪声

    '''

    # 这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  # 生成对焦矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
    k = cv2.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

    # k = k / length                         #是否归一化

    blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波

    # 转换到0-255区间
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    '''
    cv2.imshow('img',blurred)
    cv2.waitKey()
    cv2.destroyWindow('img')'''

    return blurred


class MainWidget(QWidget):

    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)

        self.__InitData()  # 先初始化数据，再初始化界面
        self.__InitView()
        self.COUNT = 0
    def __InitData(self):
        '''
                  初始化成员变量
        '''
        self.__paintBoard = PaintBoard(self)
        # 获取颜色列表(字符串类型)
        # self.__colorList = QColor.colorNames()

        self.__paintBoard.__painter = QPainter()  # 新建绘图工具
    def __InitView(self):
        '''
                  初始化界面
        '''
        self.setFixedSize(640, 480)
        self.setWindowTitle("Generate Blood Vessel image")

        # 新建一个水平布局作为本窗体的主布局
        main_layout = QHBoxLayout(self)
        # 设置主布局内边距以及控件间距为10px
        main_layout.setSpacing(10)

        # 在主界面左侧放置画板
        main_layout.addWidget(self.__paintBoard)

        # 新建垂直子布局用于放置按键
        sub_layout = QVBoxLayout()

        # 设置此子布局和内部控件的间距为10px
        sub_layout.setContentsMargins(10, 10, 10, 10)

        self.__btn_Clear = QPushButton("Clear")
        self.__btn_Clear.setParent(self)  # 设置父对象为本界面

        # 将按键按下信号与画板清空函数相关联
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear)
        sub_layout.addWidget(self.__btn_Clear)

        self.__btn_Quit = QPushButton("Exit")
        self.__btn_Quit.setParent(self)  # 设置父对象为本界面
        self.__btn_Quit.clicked.connect(self.Quit)
        sub_layout.addWidget(self.__btn_Quit)

        self.__btn_Save = QPushButton("Save")
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        sub_layout.addWidget(self.__btn_Save)

        self.__cbtn_Eraser = QCheckBox("  Erase")
        self.__cbtn_Eraser.setParent(self)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)
        sub_layout.addWidget(self.__cbtn_Eraser)

        splitter = QSplitter(self)  # 占位符
        sub_layout.addWidget(splitter)

        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("Size")
        self.__label_penThickness.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penThickness)

        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(50)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setValue(20)  # 默认粗细为10
        self.__spinBox_penThickness.setSingleStep(2)  # 最小变化值为2
        self.__spinBox_penThickness.valueChanged.connect(
            self.on_PenThicknessChange)  # 关联spinBox值变化信号和函数on_PenThicknessChange
        sub_layout.addWidget(self.__spinBox_penThickness)


        self.__btn_genBV = QPushButton("Generate PAA")
        self.__btn_genBV.setParent(self)
        self.__btn_genBV.clicked.connect(self.on_btn_genBV_Clicked)
        sub_layout.addWidget(self.__btn_genBV)


        main_layout.addLayout(sub_layout)  # 将子布局加入主布局




    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)

    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.jpg')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return
        image = self.__paintBoard.GetContentAsQImage() # 获取当前画布图像信息
        image.save(savePath[0])

    def on_btn_genBV_Clicked(self):

        qimg = self.__paintBoard.GetContentAsQImage()  # 获取当前画布图像信息
        img = QImage2CV(qimg)

        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./show_input/temp_draw_' + str(self.COUNT) + '.png', img)
        all_rain = np.zeros((256, 256), dtype=np.uint8)
        for idx in range(2):
            noise_value = np.random.randint(20, 80)
            rain_length = np.random.randint(10, 50)
            rain_angular = np.random.randint(-60, 60)
            rain_width = np.random.randint(2, 5) * 2 + 1
            noise = get_noise(img, value=noise_value)
            rain = rain_blur(noise, length=rain_length, angle=-rain_angular, w=rain_width)
            all_rain = rain + all_rain
            img = cv2.blur(img + cv2.cvtColor(rain, cv2.COLOR_GRAY2RGB), (3, 3))

        cv2.imwrite('./show_input/temp_noise_' + str(self.COUNT) + '.png', all_rain)


        # img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        cv2.imwrite('./show_input/temp_input_' + str(self.COUNT) + '.png', img)

        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        # Refesh input images of the DNN
        cv2.imwrite('./DNN_input/temp_' + str(self.COUNT) + '.png', img)
        cv2.imwrite('./DNN_input/test/input_256/temp.png', img)
        cv2.imwrite('./DNN_input/test/output_256/temp.png', img)
        ####################################################################################################################
        import torch
        import data as Data
        import model as Model
        import argparse
        import logging
        import core.logger as Logger
        import core.metrics as Metrics
        from core.wandb_logger import WandbLogger
        from tensorboardX import SummaryWriter
        import os

        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', type=str, default='config/sr_sr3.json',
                            help='JSON file for configuration')
        parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
        parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
        parser.add_argument('-debug', '-d', action='store_true')
        parser.add_argument('-enable_wandb', action='store_true')
        parser.add_argument('-log_infer', action='store_true')

        # parse configs
        args = parser.parse_args()
        opt = Logger.parse(args)
        # Convert to NoneDict, which return None for missing key.
        opt = Logger.dict_to_nonedict(opt)

        # logging
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

        Logger.setup_logger(None, opt['path']['log'],
                            'train', level=logging.INFO, screen=True)
        Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
        logger = logging.getLogger('base')
        logger.info(Logger.dict2str(opt))
        tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

        # Initialize WandbLogger
        if opt['enable_wandb']:
            wandb_logger = WandbLogger(opt)
        else:
            wandb_logger = None

        # dataset
        for phase, dataset_opt in opt['datasets'].items():
            if phase == 'val':
                val_set = Data.create_dataset(dataset_opt, phase)
                val_loader = Data.create_dataloader(
                    val_set, dataset_opt, phase)
        logger.info('Initial Dataset Finished')

        # model
        diffusion = Model.create_model(opt)
        logger.info('Initial Model Finished')

        diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['val'], schedule_phase='val')

        logger.info('Begin Model Inference.')
        current_step = 0
        current_epoch = 0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _, val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals(need_LR=False)

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]),
                        '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
        # resize the recosntructed image of graffiti && upload it to Qt surface
        sr_img = cv2.resize(sr_img, (480, 460), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./show_input/temp_sr_' + str(self.COUNT) + '.png', sr_img)

        self.singleOffset = QPoint(0, 0)  # 初始化偏移值

        self.__paintBoard.paintImg(self.COUNT)
        self.COUNT = self.COUNT + 1
    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True  # 进入橡皮擦模式
        else:
            self.__paintBoard.EraserMode = False  # 退出橡皮擦模式

    def Quit(self):
        self.close()