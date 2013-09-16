# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nuclei_options_layout.ui'
#
# Created: Thu Sep 12 16:00:43 2013
#      by: pyside-uic 0.2.13 running on PySide 1.1.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(411, 266)
        Dialog.setModal(True)
        self.label = QtGui.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(40, 30, 281, 31))
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(40, 70, 271, 17))
        self.label_2.setObjectName("label_2")
        self.groupBox = QtGui.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(60, 90, 291, 111))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.radioButton_random = QtGui.QRadioButton(self.groupBox)
        self.radioButton_random.setGeometry(QtCore.QRect(10, 40, 251, 21))
        self.radioButton_random.setObjectName("radioButton_random")
        self.radioButton_estimate = QtGui.QRadioButton(self.groupBox)
        self.radioButton_estimate.setGeometry(QtCore.QRect(10, 70, 251, 22))
        self.radioButton_estimate.setObjectName("radioButton_estimate")
        self.radioButton_no_nuclei = QtGui.QRadioButton(self.groupBox)
        self.radioButton_no_nuclei.setGeometry(QtCore.QRect(10, 10, 261, 22))
        self.radioButton_no_nuclei.setObjectName("radioButton_no_nuclei")
        self.btn_cancel = QtGui.QPushButton(Dialog)
        self.btn_cancel.setGeometry(QtCore.QRect(80, 210, 71, 27))
        self.btn_cancel.setObjectName("btn_cancel")
        self.btn_continue = QtGui.QPushButton(Dialog)
        self.btn_continue.setGeometry(QtCore.QRect(160, 210, 211, 27))
        self.btn_continue.setObjectName("btn_continue")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Input Nuclei", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", "No nuclei list .csv file was provided.", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "Continue without nuclei list?", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButton_random.setText(QtGui.QApplication.translate("Dialog", "Place random seeds", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButton_estimate.setText(QtGui.QApplication.translate("Dialog", "Estimate cell interiors (slower)", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButton_no_nuclei.setText(QtGui.QApplication.translate("Dialog", "Manually add nuclei later. (default)", None, QtGui.QApplication.UnicodeUTF8))
        self.btn_cancel.setText(QtGui.QApplication.translate("Dialog", "Cancel", None, QtGui.QApplication.UnicodeUTF8))
        self.btn_continue.setText(QtGui.QApplication.translate("Dialog", "Continue without nuclei list", None, QtGui.QApplication.UnicodeUTF8))

