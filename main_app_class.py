import sys
import cv2
from mainWindow import Ui_MainWindow
from PyQt5 import QtWidgets, uic
import pyqtgraph as pg
import numpy as np
from threading import Thread
from Helper_classes import WorkerThread, ListHandler
import logging
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph
from datetime import datetime


def create_logging_pdf(output_filename, logging_list):
    # Create a PDF document with letter size
    doc = SimpleDocTemplate(output_filename, pagesize=A4)

    # Create a custom header style
    header_style = ParagraphStyle(
        'Header1',
        parent=getSampleStyleSheet()['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        spaceAfter=10,
        textColor=colors.HexColor('#bf3043'),  # Set header color (Hex color code)
    )

    # Create a custom normal style
    normal_style = ParagraphStyle(
        'Header2',
        parent=getSampleStyleSheet()['Heading3'],
        fontName='Helvetica',
        fontSize=14,
        spaceAfter=6,
        textColor=colors.HexColor('#3366cc')  # Set header color (Hex color code)
    )

    # Create a flowable list of paragraphs for the content
    content = [Paragraph(f"<b>{item}</b>", normal_style) for item in logging_list]
    content.insert(0,
                   Paragraph(f"<b>Logging History at {datetime.now().strftime('%d/%m/%Y %H:%M')}</b> \n", header_style))

    doc.build(content)


class mainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self) -> None:
        super(mainWindow, self).__init__()
        self.setupUi(self)

        self.setWindowTitle("Mixaatoo")

        # Create a list to store log messages
        self.log_messages = []

        savePDFButton = QtWidgets.QPushButton("Save Logging History")
        self.threads_logging_layout.addWidget(savePDFButton)
        savePDFButton.clicked.connect(self.save_logging_history)

        # Create a logger and set the logging level
        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.DEBUG)

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Create a ListHandler and add it to the logger
        list_handler = ListHandler(self.log_messages)
        list_handler.setFormatter(formatter)
        self.logger.addHandler(list_handler)

        self.logger.info("Application started")

        '''
        Logging is ready and all what needed is use one of the lower functions to log: 
                self.logger.debug("This is a debug message")
                self.logger.info("This is an info message")
                self.logger.warning("This is a warning message")
                self.logger.error("This is an error message")
        '''

        ''''
        
        The lists and dictionaries of everything to avoid code relation
        
        '''''
        # this lists to store every image widget that contain the image view and ft component imageview and the combobox
        self.image_viewers = [self.imageOne, self.imageTwo, self.imageThree, self.imageFour]
        # the list that merge all sliders to be easiest to get the value
        self.sliders_weights = [self.componentOneWeight, self.componentTwoWeight, self.componentThreeWeight,
                                self.componentFourWeight]
        # the label that contain the slider value (weight of each component)
        self.sliders_labels = [self.imageOneSliderWeight, self.imageTwoSliderWeight, self.imageThreeSliderWeight,
                               self.imageFourSliderWeight]
        # the initial weight of each component that changed by slider value change
        self.weights = [1, 1, 1, 1]
        # radio button that activate each output
        self.radio_buttons = [self.activeOutputOne, self.activeOutputTwo]
        # link mode so you can only select the magnitude with phase and the real with imaginary

        self.linked_modes = {
            magnitude_np_shifted_abs: phase_np_shifted_angle,
            real_np_shifted_real: imaginary_np_shifted_imag,
            phase_np_shifted_angle: magnitude_np_shifted_abs,
            imaginary_np_shifted_imag: real_np_shifted_real
        }

        self.shifted_and_discrete_shifted_modes = {
            magnitude_np_shifted_abs: magnitude,
            phase_np_shifted_angle: phase,
            real_np_shifted_real: real,
            imaginary_np_shifted_imag: imaginary
        }

        self.worker_thread = {}
        self.cancel = False

        # this button connect with loading_output function that activate the progress bar and display output
        self.startMixingPushButton.clicked.connect(self.loading_output)

        # handel ROI changing and apply any change to every viewer
        self.handle_ROI_changing()

        # switch the flag (outer_region_selected)
        # True -> outer region
        # False -> inner region

        self.outer_region_selected = False
        self.regionSelectionComboBox.currentIndexChanged.connect(self.select_inner_or_outer_region)
        # Output views' view boxes. Use these plus addItem() to add your image
        self.add_view_box_to_the_output()

        for slider in self.sliders_weights:
            # this loop update the value of sliders if any value chane
            slider.valueChanged.connect(self.update_sliders_weights)
        for view in self.image_viewers:
            # This function linking every ROI Rect in each image viewer by the other ROI in the other viewers
            view.imageViewer.scene().sigMouseClicked.connect(self.apply_min_size_to_all_images)
        self.modeSelectionComboBox.currentIndexChanged.connect(self.setConstrains_on_comboBox)

    def setConstrains_on_comboBox(self):
        if self.modeSelectionComboBox.currentText() == magnitude_and_phase:
            print("magnitude_and_phase")
            for index, viewer in enumerate(self.image_viewers):
                viewer.FTComponentComboBox.clear()
                viewer.FTComponentComboBox.addItem(magnitude_np_shifted_abs)
                viewer.FTComponentComboBox.addItem(phase_np_shifted_angle)
        elif self.modeSelectionComboBox.currentText() == real_and_imag:
            print("real_and_imag")
            for index, viewer in enumerate(self.image_viewers):
                viewer.FTComponentComboBox.clear()
                viewer.FTComponentComboBox.addItem(real_np_shifted_real)
                viewer.FTComponentComboBox.addItem(imaginary_np_shifted_imag)

    def save_logging_history(self):
        self.logger.info("Save Logging History button was clicked")
        filepath = QtWidgets.QFileDialog.getSaveFileName(self, "Save Logging History", "loggingHistory", "PDF (*.pdf)")[
            0]
        if filepath:
            create_logging_pdf(filepath, self.log_messages)
        else:
            self.logger.warning("No file path was selected")

    def loading_output(self):
        self.cancel = False
        self.worker_thread["progressBar"] = WorkerThread()
        self.worker_thread["progressBar"].signals.update_progress.connect(self.update_progress_bar)
        self.worker_thread["progressBar"].signals.finished.connect(self.processing_finished)
        self.worker_thread["mixingImages"] = Thread(target=self.output_display_components)
        self.worker_thread["mixingImages"].start()
        self.worker_thread["progressBar"].start()
        self.startMixingPushButton.setEnabled(False)
        self.cancelMixingPushButton.setEnabled(True)
        self.cancelMixingPushButton.clicked.connect(self.cancel_processing)

    def update_progress_bar(self, value):
        self.progressBarOfThreads.setValue(value)

    def processing_finished(self):
        self.progressBarOfThreads.setValue(0)
        self.cancelMixingPushButton.setEnabled(False)
        self.startMixingPushButton.setEnabled(True)
        self.worker_thread["progressBar"].deleteLater()

    def cancel_processing(self):
        self.cancel = True
        # self.worker_thread["progressBar"].deleteLater()
        self.startMixingPushButton.setEnabled(True)
        self.cancelMixingPushButton.setEnabled(False)
        self.progressBarOfThreads.setValue(0)
        self.logger.info("Processing output cancelled")

    def add_view_box_to_the_output(self):
        self.output_one_viewer = self.outputOne.addViewBox()
        self.output_two_viewer = self.outputTwo.addViewBox()

    def handle_ROI_changing(self):
        # This function linking every ROI Rectangle in each component viewer by the other ROI in the other viewers
        self.imageOne.sig_emitter.update_ROI.connect(
            lambda: self.apply_ROI_to_all_FT_viewers(self.imageOne.roi_rectangle))
        self.imageTwo.sig_emitter.update_ROI.connect(
            lambda: self.apply_ROI_to_all_FT_viewers(self.imageTwo.roi_rectangle))
        self.imageThree.sig_emitter.update_ROI.connect(
            lambda: self.apply_ROI_to_all_FT_viewers(self.imageThree.roi_rectangle))
        self.imageFour.sig_emitter.update_ROI.connect(
            lambda: self.apply_ROI_to_all_FT_viewers(self.imageFour.roi_rectangle))

    def select_inner_or_outer_region(self):
        if self.regionSelectionComboBox.currentText() == outer_region:
            self.outer_region_selected = True
            self.logger.info("Outer region of ROI rect selected")
        elif self.regionSelectionComboBox.currentText() == inner_region:
            self.outer_region_selected = False
            self.logger.info("Inner region of ROI rect selected")

    def update_sliders_weights(self):
        for i in range(len(self.sliders_weights)):
            self.weights[i] = self.sliders_weights[i].value() / 100
            self.sliders_labels[i].setText(f"{(self.weights[i]) * 100}%")
        self.logger.info(f"Sliders have weight= {(self.sliders_weights)}")

    def calculate_mag_with_phase(self, mag, phase):
        output = np.clip(np.abs(np.fft.ifft2(np.fft.ifftshift(mag * np.exp(1j * phase)))), 0, 255)
        return output

    def calculate_real_and_imag(self, real, imag):
        output = np.clip(np.abs(np.fft.ifft2(np.fft.ifftshift(real + (1j * imag)))), 0, 255)
        return output

    def output_display_components(self):
        self.logger.info("Start mixing images")
        mode = self.modeSelectionComboBox.currentText()
        # print(mode)
        image_after_edit = self.add_the_weights_to_output_viewer(mode)

        if self.radio_buttons[0].isChecked() and not self.cancel:
            self.activate_the_output_viewer_window(image_after_edit, self.output_one_viewer)
        elif self.radio_buttons[1].isChecked() and not self.cancel:
            self.activate_the_output_viewer_window(image_after_edit, self.output_two_viewer)

    def activate_the_output_viewer_window(self, image, viewer):
        # Clear the existing items in viewer
        viewer.clear()
        # Add the modified data to viewer
        output_image_to_display = pg.ImageItem(image)
        # add image item to the viewing box
        viewer.addItem(output_image_to_display)

    def calculate_average_weight(self):
        self.averageWeight = 0
        for i in self.weights:
            if i != 0:
                self.averageWeight += 1
        self.logger.info(f"average weight of the component calculate{self.averageWeight}")

    def add_the_weights_to_output_viewer(self, output_mode):
        mag_weight = 0
        phase_weight = 0
        real_weight = 0
        imaginary_weight = 0

        for index, (viewer, weight) in enumerate(zip(self.image_viewers, self.weights)):
            mode = viewer.FTComponentComboBox.currentText()
            discrete_mode = self.shifted_and_discrete_shifted_modes[mode]

            if self.outer_region_selected == True:
                viewer.image_features_dictionary_after_any_change[the_np_array_of_image] = \
                viewer.image_features_dictionary_after_any_change[outer_region]
                viewer.calc_imag_ft(viewer.image_features_dictionary_after_any_change)
                self.logger.info("The Outer regions of the images mixed")
            else:
                viewer.image_features_dictionary_after_any_change[the_np_array_of_image] = \
                viewer.image_features_dictionary_after_any_change[inner_region]
                viewer.calc_imag_ft(viewer.image_features_dictionary_after_any_change)
                self.logger.info("The inner regions of the images mixed")

            if mode == magnitude_np_shifted_abs:
                mag_weight += viewer.image_features_dictionary_after_any_change[mode] * weight

            elif mode == phase_np_shifted_angle:
                phase_weight += viewer.image_features_dictionary_after_any_change[mode] * weight

            elif mode == real_np_shifted_real:
                real_weight += viewer.image_features_dictionary_after_any_change[mode] * weight

            elif mode == imaginary_np_shifted_imag:
                imaginary_weight += viewer.image_features_dictionary_after_any_change[mode] * weight

        if output_mode == magnitude_and_phase:
            output = self.calculate_mag_with_phase(mag_weight, phase_weight)
            self.logger.info("Mix Magnitude and Phase with the equation (Magnitude * exp(j*Phase))")
        elif output_mode == real_and_imag:
            output = self.calculate_real_and_imag(real_weight, imaginary_weight)
            self.logger.info("Mix Real and Imaginary with the equation (Real + j*Imaginary))")
        return output

    def apply_min_size_to_all_images(self):
        # get the shape of the first image
        min_height, min_width = self.image_viewers[0].image_features_dictionary[original_image_item].shape[:2]
        # apply a loop to check if there is a shape min than the first image, so i start the for loop from the second viewer
        for viewer in self.image_viewers[1:]:
            img = viewer.image_features_dictionary[original_image_item]
            # the function shape return a tuple have to dimension (height,weight)
            height, width = img.shape[:2]
            min_height = min(min_height, height)
            min_width = min(min_width, width)
        self.logger.info(f"Get the min size of the four images{min_width, min_height}")

        for i in range(len(self.image_viewers)):
            self.image_viewers[i].image_features_dictionary[original_image_item] = cv2.resize(
                self.image_viewers[i].image_features_dictionary[original_image_item], (min_width, min_height))
            new_img = self.image_viewers[i].format_image()
            new_img = cv2.cvtColor(self.image_viewers[i].image_features_dictionary[original_image_item],
                                   cv2.COLOR_BGR2GRAY)
            new_img = cv2.rotate(new_img, cv2.ROTATE_90_CLOCKWISE)
            self.image_viewers[i].add_image_data_to_the_dictionary(image=new_img)
            self.image_viewers[i].store_original_image(image=new_img)
            self.image_viewers[i].calculate_the_ft_component_of_the_dictionary(
                dictionary_before_modification=self.image_viewers[i].image_features_dictionary,
                dictionary_after_modification=self.image_viewers[i].image_features_dictionary_after_any_change)
            self.image_viewers[i].display_img(
                data=self.image_viewers[i].image_features_dictionary[the_original_image_data])
        self.logger.info(f"min size applied to all the images")

    # Function to link the ROI of all image_viewers
    def apply_ROI_to_all_FT_viewers(self, roi: pg.ROI):
        new_state = roi.getState()
        for image in self.image_viewers:
            if image.roi_rectangle is not roi:
                # Set the state of the other views without sending update signal
                image.roi_rectangle.setState(new_state, update=False)
                # Update the views after changing without sending stateChangeFinished signal
                image.roi_rectangle.stateChanged(finish=False)
                image.region_update(finish=False)


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

app = QtWidgets.QApplication(sys.argv)
window = mainWindow()
window.show()
app.exec()
