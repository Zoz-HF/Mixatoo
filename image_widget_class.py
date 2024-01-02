from PyQt5.QtWidgets import QFileDialog
from viewerwidget import Ui_image_widget_class
import pyqtgraph as pg
from PyQt5.QtGui import QCursor
from pyqtgraph.Qt import QtCore
import numpy as np
import cv2 as cv
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from signal_class import SignalEmitter
from PyQt5.QtCore import QRectF


class image_widget_class(QtWidgets.QWidget, Ui_image_widget_class):
    def __init__(self, parent: QWidget | None):
        super(image_widget_class, self).__init__(parent)
        self.setupUi(self)
        self.sig_emitter = SignalEmitter()

        self.ROI_geometry = QRectF(0, 0, 100, 100)

        self.alpha = 1
        self.beta = 1

        self.image_features_dictionary = {
            original_image_item: None, the_original_image_data: None, the_np_array_of_image: None,
            the_shifted_fft_of_the_image: None, the_fft_of_the_image: None,
            magnitude_np_shifted_abs: None, phase_np_shifted_angle: None, real_np_shifted_real: None,
            imaginary_np_shifted_imag: None, magnitude: None, phase: None,
            real: None, imaginary: None, inner_region: None, outer_region: None}

        self.image_features_dictionary_after_any_change = {
            original_image_item: None, the_original_image_data: None, the_np_array_of_image: None,
            the_shifted_fft_of_the_image: None, the_fft_of_the_image: None,
            magnitude_np_shifted_abs: None, phase_np_shifted_angle: None, real_np_shifted_real: None,
            imaginary_np_shifted_imag: None, magnitude: None, phase: None,
            real: None, imaginary: None, inner_region: None, outer_region: None}

        self.FTComponentComboBox.currentIndexChanged.connect(
            lambda: self.plot_ft_data(self.FTComponentComboBox.currentText()))

        self.image_viewer = self.imageViewer.addViewBox()
        self.viewer_constrains(self.image_viewer)

        self.ft_component_viewer = self.ft_component.addViewBox()
        self.viewer_constrains(self.ft_component_viewer)

        self.add_imageItem_to_the_viewers()

        self.initialize_ROI()

        # to update the signal and emit a new signal the flag emit signal must be TRUE
        self.roi_rectangle.sigRegionChangeFinished.connect(lambda: self.region_update(finish=True))

        # Connect the mouse click event to the function browse_img
        self.imageViewer.scene().sigMouseClicked.connect(
            lambda event: self.reset_brightness_contrast() if event.button() == 2 else None)
        self.imageViewer.scene().sigMouseClicked.connect(self.browse_image)

        # when the mouse drag the brightness and contrast change
        self.the_image_itself.mouseDragEvent = self.mouseDragEvent_to_chane_brightness_and_contrast

        # Load default image
        img_path = 'Dark_rc/init.jpg'
        self.load_image(img_path)

    def viewer_constrains(self, viewer):
        viewer.setAspectLocked(True)
        viewer.setMouseEnabled(x=False, y=False)

    def add_imageItem_to_the_viewers(self):
        self.the_image_itself = pg.ImageItem()
        self.the_image_ft_component = pg.ImageItem()
        self.image_viewer.addItem(self.the_image_itself)
        self.ft_component_viewer.addItem(self.the_image_ft_component)

    def initialize_ROI(self):
        self.roi_rectangle = pg.ROI(pos=self.ft_component_viewer.viewRect().center(), size=(200, 200), hoverPen='r',
                                    resizable=True, invertible=True, rotatable=True, maxBounds=self.ROI_geometry)
        self.roi_rectangle.setPen(color=(0, 0, 255),
                                  width=2)  # Set the pen width to make the ROI line bold (width=3, for example)
        self.ft_component_viewer.addItem(self.roi_rectangle)
        self.add_scale_handles_ROI(self.roi_rectangle)

    def region_update(self, finish=False):
        if finish:
            # Emit signal when ROI changes
            self.sig_emitter.update_ROI.emit()
        # Returns the data from the selected region mode (inner or outer)
        self.image_features_dictionary_after_any_change[outer_region], self.image_features_dictionary_after_any_change[
            inner_region] = np.fft.ifft2(np.fft.ifftshift(self.inner_outer_region_calculator()))

    def add_scale_handles_ROI(self, roi: pg.ROI):
        positions = np.array([[0, 0], [0, 0.5], [0.5, 1], [0.5, 0], [1, 0], [1, 0.5], [1, 1], [0, 1]])
        for pos in positions:
            scale_handle = roi.addScaleHandle(pos=pos, center=1 - pos)
            # Use the SizeBDiagCursor cursor to the point that stretch ROI rect
            scale_handle.setCursor(QCursor(QtCore.Qt.SizeBDiagCursor))
        # Set the ClosedHandCursor for the ROI rectangle
        roi.setCursor(QCursor(QtCore.Qt.ClosedHandCursor))

    def inner_outer_region_calculator(self):
        # Retrieve the image features from a dictionary and store them in the variable called region_data
        region_data = self.image_features_dictionary[the_shifted_fft_of_the_image]

        # This method returns a tuple containing `slice_the_data_from_array` and `QTrans`
        # `slice_the_data_from_array` likely represents a specific slice of the array
        slice_the_data_from_array, QTrans = self.roi_rectangle.getArraySlice(region_data, self.the_image_ft_component,
                                                                             returnSlice=True)

        # Create a boolean mask array of the same shape as `data` filled with `False` values.
        # This mask will be used to selectively manipulate elements in the `data` array.
        mask = np.full(region_data.shape, False)

        # True`: Set the elements in the `mask` array at positions specified by `slice_the_data_from_array` to `True`.
        # This effectively identifies a specific region in the `data` array determined by the slice indices.
        mask[slice_the_data_from_array] = True

        # Create a new array `masked_data_in` by element-wise multiplying the original `data` array with the `mask`.
        # This operation will zero out elements outside the specified region identified by the `mask`.
        masked_data_in = region_data * mask

        # Create a copy of the original `data` array and store it in `masked_data_out`.
        masked_data_out = region_data.copy()

        # remove the mask from the array and assume that this is the inner region of the ROI QRect with the values out of the ROI = zeros
        # This step effectively removes the information in the region identified by the `mask` from the copied array.
        masked_data_out[mask] = 0
        return (masked_data_in, masked_data_out)

    def browse_image(self, event):
        if event.double():
            # Open file dialog to choose an image
            file_dialog = QFileDialog(self)
            file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.tif)")
            file_dialog.setWindowTitle("Open Image File")
            file_dialog.setFileMode(QFileDialog.ExistingFile)

            if file_dialog.exec_() == QFileDialog.Accepted:
                # Get selected file path and load the image
                selected_file = file_dialog.selectedFiles()[0]
                self.load_image(selected_file)

    def format_image(self):
        image = cv.cvtColor(self.image_features_dictionary[original_image_item], cv.COLOR_BGR2GRAY)
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        return image

    def add_ROI_as_ft_component(self):
        self.image_features_dictionary_after_any_change[the_np_array_of_image] = \
        self.image_features_dictionary_after_any_change[outer_region]
        self.calc_imag_ft(self.image_features_dictionary_after_any_change)

    def load_image(self, img_path):
        self.image_features_dictionary[original_image_item] = cv.imread(img_path)
        self.image = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY)
        self.image = cv.rotate(self.image, cv.ROTATE_90_CLOCKWISE)

        self.image_features_dictionary[the_original_image_data] = self.image
        self.add_image_data_to_the_dictionary(image=self.image)

        self.calculate_the_ft_component_of_the_dictionary(dictionary_before_modification=self.image_features_dictionary,
                                                          dictionary_after_modification=self.image_features_dictionary_after_any_change)

        self.display_img(data=self.image_features_dictionary[the_np_array_of_image])
        self.ROI_geometry.adjust(0, 0, self.the_image_ft_component.width(), self.the_image_ft_component.height())

    def calc_imag_ft(self, image_dictionary):
        mode = self.FTComponentComboBox.currentText()
        # to avoid if an image don't loaded ( cannot do a non-empty take from an empty axes )
        if image_dictionary[the_np_array_of_image] is None or image_dictionary[the_np_array_of_image].size == 0:
            return

        self.store_ft_component_for_each_image(image_dictionary=image_dictionary)
        self.plot_ft_data(mode)

    def store_ft_component_for_each_image(self, image_dictionary):
        image_dictionary[the_shifted_fft_of_the_image] = self.calculate_dft_shift(
            imageDFTShift=image_dictionary[the_np_array_of_image])
        image_dictionary[the_fft_of_the_image] = self.calculate_dft(imageDFT=image_dictionary[the_np_array_of_image])

        image_dictionary[magnitude_np_shifted_abs] = self.calc_fft_mag(
            imageMag=image_dictionary[the_shifted_fft_of_the_image])
        image_dictionary[magnitude] = self.calc_fft_mag(imageMag=image_dictionary[the_fft_of_the_image])

        image_dictionary[phase_np_shifted_angle] = self.calc_fft_phase(
            imagePhase=image_dictionary[the_shifted_fft_of_the_image])
        image_dictionary[phase] = self.calc_fft_phase(imagePhase=image_dictionary[the_fft_of_the_image])

        image_dictionary[real_np_shifted_real] = self.calc_fft_real(
            imagReal=image_dictionary[the_shifted_fft_of_the_image])
        image_dictionary[real] = self.calc_fft_real(imagReal=image_dictionary[the_fft_of_the_image])

        image_dictionary[imaginary_np_shifted_imag] = self.calc_fft_imag(
            imageImag=image_dictionary[the_shifted_fft_of_the_image])
        image_dictionary[imaginary] = self.calc_fft_imag(imageImag=image_dictionary[the_fft_of_the_image])

    def calculate_dft(self, imageDFT):
        self.dft = np.fft.fft2(imageDFT)
        return self.dft

    def calculate_dft_shift(self, imageDFTShift):
        dft = self.calculate_dft(imageDFTShift)
        dft_shift = np.fft.fftshift(dft)
        return dft_shift

    def calc_fft_mag(self, imageMag):
        magnitude_shift = np.abs(imageMag)
        return magnitude_shift

    def calc_fft_phase(self, imagePhase):
        phase_shift = np.angle(imagePhase)
        return phase_shift

    def calc_fft_real(self, imagReal):
        real_shift = np.real(imagReal)
        return real_shift

    def calc_fft_imag(self, imageImag):
        imaginary_shift = np.imag(imageImag)
        return imaginary_shift

    def plot_ft_data(self, mode):
        if mode == magnitude_np_shifted_abs:
            self.the_image_ft_component.setImage(np.log(1 + self.image_features_dictionary[mode]))
        else:
            self.the_image_ft_component.setImage(self.image_features_dictionary[mode])

    def display_img(self, data):
        self.the_image_itself.setImage(data)

    def update_brightness_contrast(self):
        brightness = self.beta
        contrast = self.alpha

        # Adjust brightness and contrast
        adjusted_image = cv.convertScaleAbs(self.image_features_dictionary[the_original_image_data],
                                            alpha=(contrast / 100.0), beta=brightness)
        self.add_image_data_to_the_dictionary(image=adjusted_image)
        self.region_update(finish=True)

        self.calculate_the_ft_component_of_the_dictionary(dictionary_before_modification=self.image_features_dictionary,
                                                          dictionary_after_modification=self.image_features_dictionary_after_any_change)
        # Update the image on the viewer
        self.display_img(data=adjusted_image)

    def store_original_image(self, image):
        self.image_features_dictionary[the_original_image_data] = image
        self.image_features_dictionary_after_any_change[the_original_image_data] = image

    def calculate_the_ft_component_of_the_dictionary(self, dictionary_before_modification,
                                                     dictionary_after_modification):
        self.calc_imag_ft(dictionary_before_modification)
        self.calc_imag_ft(dictionary_after_modification)

    def add_image_data_to_the_dictionary(self, image):
        self.image_features_dictionary[the_np_array_of_image] = image
        self.image_features_dictionary_after_any_change[the_np_array_of_image] = image

    def mouseDragEvent_to_chane_brightness_and_contrast(self, event):
        # Get the drag distance from the mose event
        drag_distance = event.pos() - event.lastPos()
        # Adjust brightness and contrast based on the direction of the x-xis and y-axis
        brightness_delta = drag_distance.x() * 3
        contrast_delta = drag_distance.y() * 3
        # brightness range -127 < beta <127
        self.alpha = max(0, min(100, self.alpha + brightness_delta))
        # contrast range 0 < alpha < 1 lower contrast ....... alpha > 1 higher contrast
        self.beta = max(-127, min(100, self.beta + contrast_delta))
        # Update image with new brightness and contrast
        self.update_brightness_contrast()

    def reset_brightness_contrast(self):
        self.image_features_dictionary[the_np_array_of_image] = self.image_features_dictionary[the_original_image_data]
        self.image_features_dictionary_after_any_change[the_np_array_of_image] = self.image_features_dictionary[
            the_original_image_data]
        # if self.ft_enabled:
        #     self.region_update(finish=True)

        self.calc_imag_ft(self.image_features_dictionary)
        self.calc_imag_ft(self.image_features_dictionary_after_any_change)
        self.display_img(self.image_features_dictionary[the_original_image_data])
        # logging.debug('Reset brightness and contrast')


# Mapped global variables to store data of the np.arrays,used as keys or identifiers
# changes to variable names can be made in a centralized way by updating the string values.

# This string seems to represent the key for the original image data in dictionaries associated with image viewers.
original_image_item = "Original image"
# This string appears to represent the key for the original image data in dictionaries associated with image viewers.
the_original_image_data = " Original Img Data"
# This string appears to represent the key for the numpy array data of an image in dictionaries.
the_np_array_of_image = "Img Data"

# the shifted discrete Fourier transform (FFT) data of an image.
the_shifted_fft_of_the_image = "Shifted Discrete Image Data"
magnitude_np_shifted_abs = "FT Magnitude"
phase_np_shifted_angle = "FT Phase"
real_np_shifted_real = "FT Real"
imaginary_np_shifted_imag = "FT Imaginary"

# the discrete data to inverse the FFT
the_fft_of_the_image = "discrete fourier transform"
magnitude = "magnitude"
phase = "phase"
real = "real"
imaginary = "imaginary"

# output mode selection that merge the magnitude with phase by the equation ( Magnitude* exp(j*Phase))
magnitude_and_phase = "Magnitude/Phase"
# output mode selection that merge the Real with Imaginary by the equation ( Real * j*Imaginary)
real_and_imag = "Real/Imaginary"
#  an inner region of interest (ROI) in the image.
inner_region = "Inner region"
#  an outer region of interest (ROI) in the image.
outer_region = "Outer region"

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = image_widget_class(None)
    window.show()
    app.exec_()
