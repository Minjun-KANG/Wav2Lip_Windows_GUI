from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


#import cv2
#import threading
#import sys

# Video, Audio path get
class FaceRecognition(QDialog):

    def __init__(self, a_Video, a_Audio, a_f1_state):
        super().__init__()
        self.f1_state = a_f1_state
        self.Video_filename = a_Video
        self.Audio_filename = a_Audio
        self.setupUI()
        
    def setupUI(self):
        self.f1_state.showMessage("Face Recognition Select...")
        self.setWindowTitle("Automatically select the face of a particular person in the video.")
        self.setWindowIcon(QIcon('./resource/icon3.png'))
        self.setGeometry(200+300, 50+300, 1280-600, 960-600)

        #self.app = QApplication([])
        
        self.vbox = QVBoxLayout()
        self.combo = QComboBox(self)
        self.combo.addItem('person_01')
        self.combo.addItem('person_02')
        self.combo.activated[str].connect(self.onChanged)      
        
        self.pixmap = QPixmap("./temp/Multi/person_div.jpg")
        lbl = QLabel(self)
        lbl.setPixmap(self.pixmap)
        

        self.btn_save = QPushButton("Save and Feedback")
        self.btn_save.clicked.connect(self.onclicked)

        self.vbox.addWidget(self.combo)
        self.vbox.addWidget(lbl)
        #self.vbox.addWidget(pixmap)
        self.vbox.addWidget(self.btn_save)
        #self.app.aboutToQuit.connect(self.onExit)

        self.setLayout(self.vbox)
        self.show()

    def showModal(self):
        return super().exec_()

    def CloseModal(self):
        self.close()

    def onChanged(self, text):
        if( self.combo.currentText() == 'person_01'):
            self.pixmap = QPixmap("./temp/Multi/person_01/1.jpg")
        else:
            self.pixmap = QPixmap("./temp/Multi/person_02/1.jpg")
        self.show()
    def onclicked(self):
        self.f1_state.showMessage("Start lip-syncing. This may take more than 30 minutes... Work will be created in this folder.")