from PyQt5.QtWidgets import QFileDialog
from viewerwidget import Ui_image_widget_class
import pyqtgraph as pg
from PyQt5.QtGui import QCursor
from pyqtgraph.Qt import QtCore
import numpy as np
import cv2 as cv
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QRectF, QObject, pyqtSignal


class SignalEmitter(QObject):
    update_ROI = pyqtSignal()