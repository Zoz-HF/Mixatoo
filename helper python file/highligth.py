import sys
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QGraphicsRectItem ,QGraphicsEllipseItem
from PyQt5.QtGui import QPixmap, QPainter, QColor, QImage
from PyQt5.QtCore import Qt, QRectF

class ImageViewer(QGraphicsView):
    def __init__(self):
        super(ImageViewer, self).__init__()

        # Create a QGraphicsScene and set it as the scene for the view
        scene = QGraphicsScene(self)
        self.setScene(scene)

        # Load an image and create a QGraphicsPixmapItem to display it
        image_path = "image 1.jpg"  # Replace with the path to your image
        self.original_pixmap = QPixmap(image_path)
        self.grayscale_pixmap = self.convert_to_grayscale(self.original_pixmap)
        self.pixmap_item = QGraphicsPixmapItem(self.grayscale_pixmap)
        scene.addItem(self.pixmap_item)

        # Variables to track the rectangle drawing
        self.drawing_rectangle = False
        self.rectangle_item = None
        self.start_point = None

    def convert_to_grayscale(self, pixmap):
        image = pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
        grayscale_pixmap = QPixmap.fromImage(image)
        return grayscale_pixmap

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.start_point = event.pos()
            self.drawing_rectangle = True

    def mouseMoveEvent(self, event):
        if self.drawing_rectangle:
            current_point = event.pos()
            rect = QRectF(self.start_point, current_point).normalized()

            if not self.rectangle_item:
                print("yes")
                self.rectangle_item = QGraphicsRectItem(rect)
                self.rectangle_item.setPen(QColor(Qt.blue))
                self.scene().addItem(self.rectangle_item)
            else:
                self.rectangle_item.setRect(rect)
                print("no")

    def mouseReleaseEvent(self, event):
        if self.drawing_rectangle:
            self.drawing_rectangle = False
            self.start_point = None

            # Get information about the points inside the rectangle
            points_in_rectangle = self.get_points_in_rectangle(self.rectangle_item.rect())
            # print("Points inside the rectangle:", points_in_rectangle)

    def get_points_in_rectangle(self, rectangle):
        # Get the region of interest from the image
        region_of_interest = self.grayscale_pixmap.toImage().copy(rectangle.toRect()).convertToFormat(QImage.Format_Grayscale8)
        # Get the pixel values within the rectangle
        points = []
        for y in range(region_of_interest.height()):
            for x in range(region_of_interest.width()):
                pixel_value = region_of_interest.pixel(x, y)
                points.append((x + int(rectangle.x()), y + int(rectangle.y()), QColor(pixel_value).getRgb()))
        return points

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
