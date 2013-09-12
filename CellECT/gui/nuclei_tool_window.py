# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nuclei_tool_layout.ui'
#
# Created: Wed Aug 28 23:42:33 2013
#      by: pyside-uic 0.2.13 running on PySide 1.1.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(531, 653)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 40, 461, 441))
        self.label.setAutoFillBackground(True)
        self.label.setText("")
        self.label.setObjectName("label")
        self.horizontalSlider = QtGui.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(90, 504, 421, 29))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider_2 = QtGui.QSlider(self.centralwidget)
        self.horizontalSlider_2.setGeometry(QtCore.QRect(90, 534, 421, 29))
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 511, 81, 17))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 540, 81, 17))
        self.label_3.setObjectName("label_3")
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(30, 570, 141, 27))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(190, 570, 141, 27))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(350, 570, 141, 27))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 10, 421, 17))
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 531, 25))
        self.menubar.setObjectName("menubar")
        self.menuNuclei_Tool = QtGui.QMenu(self.menubar)
        self.menuNuclei_Tool.setObjectName("menuNuclei_Tool")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionSave_Seeds = QtGui.QAction(MainWindow)
        self.actionSave_Seeds.setObjectName("actionSave_Seeds")
        self.actionUndo_Last = QtGui.QAction(MainWindow)
        self.actionUndo_Last.setObjectName("actionUndo_Last")
        self.actionClose = QtGui.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.menuNuclei_Tool.addAction(self.actionSave_Seeds)
        self.menuNuclei_Tool.addAction(self.actionUndo_Last)
        self.menuNuclei_Tool.addAction(self.actionClose)
        self.menubar.addAction(self.menuNuclei_Tool.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "Initial Seeds (Nuclei) Tool", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("MainWindow", "Navigate z:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("MainWindow", "Navigate t:", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton.setText(QtGui.QApplication.translate("MainWindow", "Finish workspace", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_2.setText(QtGui.QApplication.translate("MainWindow", "Undo last seed", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_3.setText(QtGui.QApplication.translate("MainWindow", "Cancel", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("MainWindow", "Add seeds for every time stamp to complete workspace:", None, QtGui.QApplication.UnicodeUTF8))
        self.menuNuclei_Tool.setTitle(QtGui.QApplication.translate("MainWindow", "Nuclei Tool", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSave_Seeds.setText(QtGui.QApplication.translate("MainWindow", "Save Seeds", None, QtGui.QApplication.UnicodeUTF8))
        self.actionUndo_Last.setText(QtGui.QApplication.translate("MainWindow", "Undo Last", None, QtGui.QApplication.UnicodeUTF8))
        self.actionClose.setText(QtGui.QApplication.translate("MainWindow", "Close", None, QtGui.QApplication.UnicodeUTF8))

