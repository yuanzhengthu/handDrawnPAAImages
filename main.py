'''
created on 2022-08-25
author: yuanthu
description: painting blood vessel
'''

from MainWidget import MainWidget
from PyQt5.QtWidgets import QApplication
import sys

def main():
    app = QApplication(sys.argv)
    mainWidget = MainWidget()  # 新建主界面
    mainWidget.show()  # 显示主界面
    exit(app.exec_())  # 进入消息循环
if __name__ == '__main__':
    main()