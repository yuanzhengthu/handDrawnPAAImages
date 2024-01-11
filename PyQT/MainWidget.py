'''
Created on 2022-08-25
Author: yuanthu
Description: Painting blood vessels
'''

from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox, QPoint, qRed, qBlue, qGreen, QPainter, QImage, QRect
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QSplitter, QComboBox, QLabel, QSpinBox, QFileDialog
from PyQT.PaintBoard import PaintBoard
import cv2
import numpy as np
from data.HR_dataset import HRDataset
import data as Data
from model.model import DDPM
from utils import parse, tensor2img
import argparse

def QImage2CV(qimg):
    """
    Convert QImage to OpenCV format.
    """
    tmp = qimg
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
    """
    Generate noisy image.
    """
    noise = np.random.uniform(0, 256, img.shape[0:2])
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0
    k = np.array([[0, 0.1, 0], [0.1, 8, 0.1], [0, 0.1, 0]])
    noise = cv2.filter2D(noise, -1, k)

    return noise


def rain_blur(noise, length=10, angle=0, w=1):
    """
    Apply rain blur effect.
    """
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))
    k = cv2.warpAffine(dig, trans, (length, length))
    k = cv2.GaussianBlur(k, (w, w), 0)
    blurred = cv2.filter2D(noise, -1, k)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    return blurred


class MainWidget(QWidget):
    def __init__(self, Parent=None):
        """
        Constructor
        """
        super().__init__(Parent)
        self.__InitData()
        self.__InitView()
        self.COUNT = 0

    def __InitData(self):
        """
        Initialize member variables.
        """
        self.__paintBoard = PaintBoard(self)
        self.__paintBoard.__painter = QPainter()

    def __InitView(self):
        """
        Initialize the user interface.
        """
        self.setFixedSize(640, 480)
        self.setWindowTitle("Generate Blood Vessel Image")

        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)

        main_layout.addWidget(self.__paintBoard)

        sub_layout = QVBoxLayout()
        sub_layout.setContentsMargins(10, 10, 10, 10)

        self.__btn_Clear = QPushButton("Clear")
        self.__btn_Clear.setParent(self)
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear)
        sub_layout.addWidget(self.__btn_Clear)

        self.__btn_Quit = QPushButton("Exit")
        self.__btn_Quit.setParent(self)
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

        splitter = QSplitter(self)
        sub_layout.addWidget(splitter)

        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("Size")
        self.__label_penThickness.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penThickness)

        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(50)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setValue(20)
        self.__spinBox_penThickness.setSingleStep(2)
        self.__spinBox_penThickness.valueChanged.connect(self.on_PenThicknessChange)
        sub_layout.addWidget(self.__spinBox_penThickness)

        self.__btn_genBV = QPushButton("Generate PAA")
        self.__btn_genBV.setParent(self)
        self.__btn_genBV.clicked.connect(self.on_btn_genBV_Clicked)
        sub_layout.addWidget(self.__btn_genBV)

        main_layout.addLayout(sub_layout)

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)

    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.jpg')
        if savePath[0] == "":
            print("Save canceled")
            return
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath[0])

    def on_btn_genBV_Clicked(self):
        qimg = self.__paintBoard.GetContentAsQImage()
        img = QImage2CV(qimg)

        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./PyQT/qt_showoff/temp_draw_' + str(self.COUNT) + '.png', img)
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

        cv2.imwrite('./PyQT/qt_showoff/temp_noise_' + str(self.COUNT) + '.png', all_rain)
        cv2.imwrite('./PyQT/qt_showoff/temp_input_' + str(self.COUNT) + '.png', img)

        img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./PyQT/qt_input/temp_' + str(self.COUNT) + '.png', img)
        cv2.imwrite('./PyQT/qt_input/test/input_256/temp.png', img)
        cv2.imwrite('./PyQT/qt_input/test/output_256/temp.png', img)

        # Generate photoacoustic image with diffusion model
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', type=str, default='image_generation.json', help='JSON file for configuration')
        parser.add_argument('-p', '--phase', type=str, choices=['train'], help='train', default='train')
        parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
        parser.add_argument('-debug', '-d', action='store_true')
        parser.add_argument('-enable_wandb', action='store_true')
        parser.add_argument('-log_infer', action='store_true')
        args = parser.parse_args()
        opt = parse(args)

        # Initialize dataset and dataloader
        opt["path"]["resume_state"] = "experiments/PA"
        opt["datasets"]["val"]["dataroot"] = "PyQT/qt_input"
        opt["datasets"]["val"]["mode"] = "HR"
        opt["datasets"]["val"]["l_resolution"] = 50
        opt["datasets"]["val"]["r_resolution"] = 400
        val_set = HRDataset(opt["datasets"]["val"], 'val')
        val_loader = Data.create_dataloader(val_set, opt["datasets"]["val"], 'val')
        # Initialize the diffusion model
        diffusion = DDPM(opt)
        # Set the noise schedule
        diffusion.set_new_noise_schedule()
        for idx, val_data in enumerate(val_loader):
            diffusion.feed_data(val_data)
            diffusion.test(continous=False)  # You may need to adjust the test function based on your model
            visuals = diffusion.get_current_visuals(need_LR=False)
            forged_img = tensor2img(visuals['SR'])
            forged_img = cv2.resize(forged_img, (480, 460), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('PyQT/qt_showoff/temp_sr_' + str(self.COUNT) + '.png', forged_img)

        self.singleOffset = QPoint(0, 0)
        self.__paintBoard.paintImg(self.COUNT)
        self.COUNT = self.COUNT + 1

    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True
        else:
            self.__paintBoard.EraserMode = False

    def Quit(self):
        self.close()
