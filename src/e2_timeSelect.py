from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import subprocess
#import cv2
#import threading
#import sys

# Video, Audio path get

class TimeSelect(QDialog):
    
    def __init__(self, a_Video, a_Audio, a_f1_state):
        self.start_time = None
        self.end_time = None
        self.save_path = None
        super().__init__()
        #self.total_time = a_time
        self.f1_state = a_f1_state
        self.Video_filename = a_Video
        self.Audio_filename = a_Audio
        
        self.setupUI()
        
    def setupUI(self):
        self.f1_state.showMessage("Time Select...")
        self.setWindowTitle("Select a time zone for the video")
        self.setWindowIcon(QIcon('./resource/icon3.png'))
        self.setGeometry(200+300, 50+300, 1280-900, 960-800)


        self.t1_1_StartLabel = QLabel("Start Time")
        self.t1_2_StartTimeEdit = QTimeEdit(self)
        self.t1_2_StartTimeEdit.setTime(QTime(00,00,00))
        self.t1_2_StartTimeEdit.setTimeRange(QTime(00,00,00), QTime(23,59,59))
        self.t1_2_StartTimeEdit.setDisplayFormat('hh:mm:ss')

        self.t2_1_EndLabel = QLabel("End  Time")
        self.t2_2_EndTimeEdit = QTimeEdit(self)
        self.t2_2_EndTimeEdit.setTime(QTime(00,00,3))
        self.t2_2_EndTimeEdit.setTimeRange(QTime(00,00,00), QTime(23,59,59))
        self.t2_2_EndTimeEdit.setDisplayFormat('hh:mm:ss')


        self.t3_1_SaveBtn = QPushButton("Save and Feedback")
        self.t3_1_SaveBtn.clicked.connect(self.exec_save)
        

        self.t_layout = QGridLayout()
        self.t_layout.addWidget(self.t1_1_StartLabel, 0, 0)
        self.t_layout.addWidget(self.t1_2_StartTimeEdit, 0, 1)
        self.t_layout.addWidget(self.t2_1_EndLabel, 1, 0)
        self.t_layout.addWidget(self.t2_2_EndTimeEdit, 1, 1)
        self.t_layout.addWidget(self.t3_1_SaveBtn, 2, 0, 1, 2)

       

        self.setLayout(self.t_layout)
    
    def exec_save(self):
        #print(str(self.t1_2_StartTimeEdit.time().toString("H:m:s")))
        self.start_time = str(self.t1_2_StartTimeEdit.time().toString("HH:mm:ss"))
        #print(str(self.t2_2_EndTimeEdit.time().toString("H:m:s")))
        self.end_time = str(self.t2_2_EndTimeEdit.time().toString("HH:mm:ss"))
        self.clip_video()


    def showModal(self):
        return super().exec_()

    def CloseModal(self):
        self.close()

    def clip_video(self):
        print("clip run")
        #self.save_path = './src/temp/crop_video.avi'
        self.save_path = './crop_video.avi'
        command = 'ffmpeg -y -i {} -ss {} -to {} {}'.format(self.Video_filename, self.start_time, self.end_time, self.save_path)
        print(command)
        subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("clip done")
        self.CloseModal()
        

    def get_filename(self):
        return self.save_path

