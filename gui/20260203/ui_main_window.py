# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_windowoEXARV.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QGridLayout, QHBoxLayout,
    QHeaderView, QMainWindow, QMenuBar, QPushButton,
    QSizePolicy, QSplitter, QStatusBar, QTabWidget,
    QTableView, QTreeView, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(746, 593)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.splitter_2 = QSplitter(self.centralwidget)
        self.splitter_2.setObjectName(u"splitter_2")
        self.splitter_2.setOrientation(Qt.Orientation.Horizontal)
        self.layoutWidget = QWidget(self.splitter_2)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.verticalLayout = QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.checkBox_png = QCheckBox(self.layoutWidget)
        self.checkBox_png.setObjectName(u"checkBox_png")

        self.gridLayout.addWidget(self.checkBox_png, 1, 0, 1, 1)

        self.checkBox_bmp = QCheckBox(self.layoutWidget)
        self.checkBox_bmp.setObjectName(u"checkBox_bmp")

        self.gridLayout.addWidget(self.checkBox_bmp, 1, 1, 1, 1)

        self.checkBox_jpg = QCheckBox(self.layoutWidget)
        self.checkBox_jpg.setObjectName(u"checkBox_jpg")

        self.gridLayout.addWidget(self.checkBox_jpg, 1, 2, 1, 1)

        self.pushButton_refresh = QPushButton(self.layoutWidget)
        self.pushButton_refresh.setObjectName(u"pushButton_refresh")

        self.gridLayout.addWidget(self.pushButton_refresh, 0, 2, 1, 1)

        self.pushButton_open = QPushButton(self.layoutWidget)
        self.pushButton_open.setObjectName(u"pushButton_open")

        self.gridLayout.addWidget(self.pushButton_open, 0, 0, 1, 2)


        self.verticalLayout.addLayout(self.gridLayout)

        self.splitter = QSplitter(self.layoutWidget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setMinimumSize(QSize(200, 0))
        self.splitter.setOrientation(Qt.Orientation.Vertical)
        self.treeView = QTreeView(self.splitter)
        self.treeView.setObjectName(u"treeView")
        self.treeView.setMinimumSize(QSize(0, 150))
        self.treeView.setBaseSize(QSize(0, 0))
        self.splitter.addWidget(self.treeView)
        self.tableView = QTableView(self.splitter)
        self.tableView.setObjectName(u"tableView")
        self.tableView.setMinimumSize(QSize(0, 300))
        self.splitter.addWidget(self.tableView)

        self.verticalLayout.addWidget(self.splitter)

        self.splitter_2.addWidget(self.layoutWidget)
        self.tabWidget = QTabWidget(self.splitter_2)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setMinimumSize(QSize(500, 0))
        self.tab_image = QWidget()
        self.tab_image.setObjectName(u"tab_image")
        self.gridLayout_2 = QGridLayout(self.tab_image)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.widget_image = QWidget(self.tab_image)
        self.widget_image.setObjectName(u"widget_image")

        self.gridLayout_2.addWidget(self.widget_image, 0, 0, 1, 1)

        self.tabWidget.addTab(self.tab_image, "")
        self.tab_histogram = QWidget()
        self.tab_histogram.setObjectName(u"tab_histogram")
        self.gridLayout_3 = QGridLayout(self.tab_histogram)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.widget_histogram = QWidget(self.tab_histogram)
        self.widget_histogram.setObjectName(u"widget_histogram")

        self.gridLayout_3.addWidget(self.widget_histogram, 0, 0, 1, 1)

        self.tabWidget.addTab(self.tab_histogram, "")
        self.splitter_2.addWidget(self.tabWidget)

        self.horizontalLayout.addWidget(self.splitter_2)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 746, 33))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.checkBox_png.setText(QCoreApplication.translate("MainWindow", u"*.png", None))
        self.checkBox_bmp.setText(QCoreApplication.translate("MainWindow", u"*.bmp", None))
        self.checkBox_jpg.setText(QCoreApplication.translate("MainWindow", u"*.jpg", None))
        self.pushButton_refresh.setText(QCoreApplication.translate("MainWindow", u"Refresh", None))
        self.pushButton_open.setText(QCoreApplication.translate("MainWindow", u"Open", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_image), QCoreApplication.translate("MainWindow", u"Image", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_histogram), QCoreApplication.translate("MainWindow", u"Histogram", None))
    # retranslateUi

