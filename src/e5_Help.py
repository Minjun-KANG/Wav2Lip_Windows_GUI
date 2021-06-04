from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


#import cv2
#import threading
#import sys

# Video, Audio path get
class Help(QDialog):

    def __init__(self, a_Video, a_Audio, a_f1_state):
        super().__init__()
        self.f1_state = a_f1_state
        self.Video_filename = a_Video
        self.Audio_filename = a_Audio
        self.setupUI()
        
    def setupUI(self):
        
        self.setWindowTitle("Help")
        self.setWindowIcon(QIcon('./resource/icon3.png'))
        self.setGeometry(200+150, 50+150, 1280-300, 960-300)

        self.k1 = QLabel("1. 비디오와 오디오를 선택합니다.")
        self.k2 = QLabel("비디오는 AVI로 작업하는 것을 권장합니다.(mp4, K-Lite codec 필요)")
        self.k3 = QLabel("")
        self.k4 = QLabel("2. Lip sync 버튼을 통해 Video의 인물 입모양을 Audio에 맞게 립싱크 합니다.")
        self.k5 = QLabel("단, 인물이 한명만 등장하며, Video가 말하는 부분에 맞춰 Audio를 따로 녹음해야합니다.")
        self.k6 = QLabel("")
        self.k7 = QLabel("3. 비디오의 시간을 선택하고 싶은 경우")
        self.k8 = QLabel("Time Select 버튼을 통해 시작 시각과 끝 시각을 지정하여 자릅니다.")
        self.k9 = QLabel("단, 바로 Lip sync 주 영상이 바뀝니다.")
        self.k10 = QLabel("")
        self.k11 = QLabel("4. 영상의 여러 인물이 등장하는 경우")
        self.k12 = QLabel("Face Recognition 버튼을 통해 여러 인물 중 하나의 인물을 선택하세요.")
        self.k13 = QLabel("선택한 인물의 얼굴 픽셀만을 추출합니다.")
        self.k14 = QLabel("단 바로 Lip sync 주 영상이 바뀝니다. ")
        self.k15 = QLabel("")


        self.e1 = QLabel("1. Select Video and Audio.")
        self.e2 = QLabel("Video is recommended to work with AVI. (mp4, K-Lite codec required)")
        self.e3 = QLabel("")
        self.e4 = QLabel("2. Lip sync button to lip sync the video's character mouth shape to the audio.")
        self.e5 = QLabel("However, only one person appears, and audio should be recorded separately according to what the video says.")
        self.e6 = QLabel("")
        self.e7 = QLabel("3. If you want to select a time for your video")
        self.e8 = QLabel("Use the Time Select button to specify a start and end time to crop.")
        self.e9 = QLabel("However, the Lip sync main video changes immediately.")
        self.e10 = QLabel("")
        self.e11 = QLabel("4. When multiple people appear in the video")
        self.e12 = QLabel("Use the Face Recognition button to select a person from multiple people.")
        self.e13 = QLabel("Extracts only the face pixels of the selected person.")
        self.e14 = QLabel("However, the Lip sync main video changes immediately. ")
        self.e15 = QLabel("")

        



        self.korean_layout = QVBoxLayout()
        self.korean_layout.addStretch(1)
        self.korean_layout.addWidget(self.k1)
        self.korean_layout.addWidget(self.k2)
        self.korean_layout.addWidget(self.k3)
        self.korean_layout.addWidget(self.k4)
        self.korean_layout.addWidget(self.k5)
        self.korean_layout.addWidget(self.k6)
        self.korean_layout.addWidget(self.k7)
        self.korean_layout.addWidget(self.k8)
        self.korean_layout.addWidget(self.k9)
        self.korean_layout.addWidget(self.k10)
        self.korean_layout.addWidget(self.k11)
        self.korean_layout.addWidget(self.k12)
        self.korean_layout.addWidget(self.k13)
        self.korean_layout.addWidget(self.k14)
        self.korean_layout.addWidget(self.k15)
        self.korean_layout.addStretch(1)

        
        self.english_layout = QVBoxLayout()
        self.english_layout.addStretch(1)
        self.english_layout.addWidget(self.e1)
        self.english_layout.addWidget(self.e2)
        self.english_layout.addWidget(self.e3)
        self.english_layout.addWidget(self.e4)
        self.english_layout.addWidget(self.e5)
        self.english_layout.addWidget(self.e6)
        self.english_layout.addWidget(self.e7)
        self.english_layout.addWidget(self.e8)
        self.english_layout.addWidget(self.e9)
        self.english_layout.addWidget(self.e10)
        self.english_layout.addWidget(self.e11)
        self.english_layout.addWidget(self.e12)
        self.english_layout.addWidget(self.e13)
        self.english_layout.addWidget(self.e14)
        self.english_layout.addWidget(self.e15)
        self.english_layout.addStretch(1)
        
        self.h_layout = QGridLayout()
        self.h_layout.addLayout(self.korean_layout, 0,0)
        
        self.h_layout.addLayout(self.english_layout, 0,1)

       
        self.setLayout(self.h_layout)

    def showModal(self):
        return super().exec_()



