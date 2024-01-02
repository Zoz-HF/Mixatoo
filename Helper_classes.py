import logging
from PyQt5.QtCore import QThread, pyqtSignal, QObject


class WorkerSignals(QObject):
    finished = pyqtSignal()
    update_progress = pyqtSignal(int)


class WorkerThread(QThread):
    def __init__(self):
        super(WorkerThread, self).__init__()
        self.signals = WorkerSignals()

    def run(self):
        for i in range(1, 101, 5):
            self.msleep(1)  # Simulate work
            self.signals.update_progress.emit(i)
        self.signals.finished.emit()


class ListHandler(logging.Handler):
    def __init__(self, log_list):
        super().__init__()
        self.log_list = log_list

    def emit(self, record):
        log_message = self.format(record)
        self.log_list.append(log_message)
