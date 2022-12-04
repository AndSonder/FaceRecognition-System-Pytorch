from gui.faceWindow import FaceWindow
from PySide6.QtWidgets import QApplication


def main():
    app = QApplication([])
    window = FaceWindow()
    app.exec_()

if __name__ == '__main__':
    main()