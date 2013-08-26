# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'about_layout.ui'
#
# Created: Sat Aug 17 16:21:51 2013
#      by: pyside-uic 0.2.13 running on PySide 1.1.0
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        Dialog.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        Dialog.setModal(False)
        self.label = QtGui.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(30, 40, 341, 161))
        self.label.setObjectName("label")
        self.pushButton = QtGui.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(280, 250, 97, 31))
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "About CellECT", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", "ABout CellECT", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton.setText(QtGui.QApplication.translate("Dialog", "KTnxBye", None, QtGui.QApplication.UnicodeUTF8))

