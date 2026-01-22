# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'oled_analyzerWLNwso.ui'
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
from PySide6.QtWidgets import (QApplication, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QMenuBar,
    QSizePolicy, QSlider, QSpinBox, QSplitter,
    QStatusBar, QTabWidget, QTreeView, QVBoxLayout,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(640, 480)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.leftWidget = QWidget(self.splitter)
        self.leftWidget.setObjectName(u"leftWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.leftWidget.sizePolicy().hasHeightForWidth())
        self.leftWidget.setSizePolicy(sizePolicy)
        self.leftWidget.setMinimumSize(QSize(200, 400))
        self.horizontalLayout_2 = QHBoxLayout(self.leftWidget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.fileTreeView = QTreeView(self.leftWidget)
        self.fileTreeView.setObjectName(u"fileTreeView")

        self.horizontalLayout_2.addWidget(self.fileTreeView)

        self.splitter.addWidget(self.leftWidget)
        self.rightWidget = QWidget(self.splitter)
        self.rightWidget.setObjectName(u"rightWidget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.rightWidget.sizePolicy().hasHeightForWidth())
        self.rightWidget.setSizePolicy(sizePolicy1)
        self.rightWidget.setMinimumSize(QSize(600, 400))
        self.horizontalLayout_3 = QHBoxLayout(self.rightWidget)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.mainTabWidget = QTabWidget(self.rightWidget)
        self.mainTabWidget.setObjectName(u"mainTabWidget")
        self.tab1 = QWidget()
        self.tab1.setObjectName(u"tab1")
        self.verticalLayout_2 = QVBoxLayout(self.tab1)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.rgbCanvas = QWidget(self.tab1)
        self.rgbCanvas.setObjectName(u"rgbCanvas")

        self.verticalLayout_2.addWidget(self.rgbCanvas)

        self.mainTabWidget.addTab(self.tab1, "")
        self.tab2 = QWidget()
        self.tab2.setObjectName(u"tab2")
        self.lumCanvas = QWidget(self.tab2)
        self.lumCanvas.setObjectName(u"lumCanvas")
        self.lumCanvas.setGeometry(QRect(9, 9, 380, 94))
        self.verticalLayout_3 = QVBoxLayout(self.lumCanvas)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.groupBox = QGroupBox(self.lumCanvas)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.meanSpinBox = QSpinBox(self.groupBox)
        self.meanSpinBox.setObjectName(u"meanSpinBox")
        self.meanSpinBox.setMaximum(255)
        self.meanSpinBox.setValue(128)

        self.gridLayout.addWidget(self.meanSpinBox, 0, 1, 2, 1)

        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)

        self.stdSpinBox = QSpinBox(self.groupBox)
        self.stdSpinBox.setObjectName(u"stdSpinBox")
        self.stdSpinBox.setMinimum(1)
        self.stdSpinBox.setMaximum(100)
        self.stdSpinBox.setValue(20)

        self.gridLayout.addWidget(self.stdSpinBox, 0, 3, 2, 1)

        self.meanSlider = QSlider(self.groupBox)
        self.meanSlider.setObjectName(u"meanSlider")
        self.meanSlider.setMaximum(255)
        self.meanSlider.setValue(128)
        self.meanSlider.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout.addWidget(self.meanSlider, 1, 0, 1, 1)

        self.stdSlider = QSlider(self.groupBox)
        self.stdSlider.setObjectName(u"stdSlider")
        self.stdSlider.setMaximum(100)
        self.stdSlider.setValue(20)
        self.stdSlider.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout.addWidget(self.stdSlider, 1, 2, 1, 1)


        self.verticalLayout_3.addWidget(self.groupBox)

        self.mainTabWidget.addTab(self.tab2, "")
        self.tab3 = QWidget()
        self.tab3.setObjectName(u"tab3")
        self.verticalLayout_4 = QVBoxLayout(self.tab3)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.histCanvas = QWidget(self.tab3)
        self.histCanvas.setObjectName(u"histCanvas")

        self.verticalLayout_4.addWidget(self.histCanvas)

        self.mainTabWidget.addTab(self.tab3, "")
        self.tab4 = QWidget()
        self.tab4.setObjectName(u"tab4")
        self.verticalLayout_5 = QVBoxLayout(self.tab4)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.anomalyCanvas = QWidget(self.tab4)
        self.anomalyCanvas.setObjectName(u"anomalyCanvas")
        self.groupBox_2 = QGroupBox(self.anomalyCanvas)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(10, 20, 189, 76))
        self.gridLayout_2 = QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label_3 = QLabel(self.groupBox_2)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout_2.addWidget(self.label_3, 0, 0, 1, 1)

        self.thresholdSpinBox = QSpinBox(self.groupBox_2)
        self.thresholdSpinBox.setObjectName(u"thresholdSpinBox")
        self.thresholdSpinBox.setMaximum(100)
        self.thresholdSpinBox.setValue(50)

        self.gridLayout_2.addWidget(self.thresholdSpinBox, 0, 1, 2, 1)

        self.thresholdSlider = QSlider(self.groupBox_2)
        self.thresholdSlider.setObjectName(u"thresholdSlider")
        self.thresholdSlider.setMaximum(100)
        self.thresholdSlider.setValue(50)
        self.thresholdSlider.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_2.addWidget(self.thresholdSlider, 1, 0, 1, 1)


        self.verticalLayout_5.addWidget(self.anomalyCanvas)

        self.mainTabWidget.addTab(self.tab4, "")

        self.horizontalLayout_3.addWidget(self.mainTabWidget)

        self.splitter.addWidget(self.rightWidget)

        self.verticalLayout.addWidget(self.splitter)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 640, 33))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.mainTabWidget.setCurrentIndex(3)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.mainTabWidget.setTabText(self.mainTabWidget.indexOf(self.tab1), QCoreApplication.translate("MainWindow", u"RGB", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Parameter", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Target Mean", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Target Std", None))
        self.mainTabWidget.setTabText(self.mainTabWidget.indexOf(self.tab2), QCoreApplication.translate("MainWindow", u"Luminance", None))
        self.mainTabWidget.setTabText(self.mainTabWidget.indexOf(self.tab3), QCoreApplication.translate("MainWindow", u"Histogram", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Threshold Setting", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Threshold", None))
        self.mainTabWidget.setTabText(self.mainTabWidget.indexOf(self.tab4), QCoreApplication.translate("MainWindow", u"Anomaly Map", None))
    # retranslateUi

