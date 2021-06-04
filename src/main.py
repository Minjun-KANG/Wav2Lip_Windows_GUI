import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from cv2 import *
from moviepy.video.io.VideoFileClip import VideoFileClip

#waveform 출력
from scipy.io import wavfile
import moviepy.editor as mp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

#wav2lip 실행
import subprocess
import threading

#pre, post processing dialog
from e2_timeSelect import TimeSelect
from e3_faceRecognition import FaceRecognition
from e5_Help import Help



class ThreadClass(threading.Thread): 
    Video_filename = None
    Audio_filename = None
    def __init__(self, a_video, a_audio, parent = None): 
        self.Video_filename = a_video
        self.Audio_filename = a_audio
        super(ThreadClass,self).__init__(parent)
    def run(self):
        import Wav2Lip
        Wav2Lip.inference.main(self.Video_filename, self.Audio_filename)

    # def croprun(self):
    #     dlg = roi.staticROI(self.Video_filename)
    #     #dlg = VideoCropDialog.VideoROIDialog(self.Video_filename)
    #     dlg.exec_()



class MyWindow(QWidget):
    Video_filename = None
    Audio_filename = None
    def __init__(self):
        super().__init__()
        self.setupUI()
        

    def setupUI(self):
        self.finalFrame = -1
        self.setGeometry(200, 50, 1280, 960)
        self.setWindowTitle("Y2X")
        self.setWindowIcon(QIcon('./resource/icon3.png'))
        btnSize = QSize(100,20) # 공통된 버튼 사이즈

        #last layout
        #파일 추가 버튼 설정
        #비디오파일 추가 버튼
        self.a1_1_OpenVideoBtn = QPushButton("Open Video")
        self.a1_1_OpenVideoBtn.setMaximumWidth(100) # 공통된 그리드 사이즈
        self.a1_1_OpenVideoBtn.setToolTip("Open Video File")
        self.a1_1_OpenVideoBtn.setStatusTip("Open Video File")
        self.a1_1_OpenVideoBtn.setFixedHeight(24)
        self.a1_1_OpenVideoBtn.setIconSize(btnSize)
        self.a1_1_OpenVideoBtn.clicked.connect(self.VideoOpen)
        #업로드 된 파일네임 출력 영역
        self.a1_2_VideoFileName = QStatusBar()
        self.a1_2_VideoFileName.setFixedHeight(14)
        self.a1_2_VideoFileName.showMessage("Please upload video (.mp4, .avi)")



        #오디오파일 추가 버튼
        self.a2_1_OpenAudioBtn = QPushButton("Open Audio")
        self.a2_1_OpenAudioBtn.setToolTip("Open Video File")
        self.a2_1_OpenAudioBtn.setStatusTip("Open Video File")
        self.a2_1_OpenAudioBtn.setFixedHeight(24)
        self.a2_1_OpenAudioBtn.setIconSize(btnSize)
        self.a2_1_OpenAudioBtn.clicked.connect(self.AudioOpen)
        self.a2_2_AudioFileName = QStatusBar()
        self.a2_2_AudioFileName.setFixedHeight(14)
        self.a2_2_AudioFileName.showMessage("Please upload audio (.mp3, .wav)")

        self.a_3_1_Btn = QPushButton("Open Audio")
        self.a_3_2_Btn = QLabel("Hello")
        
        
        #비디오 출력 영역
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        b1_videoWidget = QVideoWidget()
        self.mediaPlayer.setVideoOutput(b1_videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        b1_videoWidget.setFixedHeight(480*1.2)
        b1_videoWidget.setFixedWidth(640*1.2)
        b1_videoWidget.setStyleSheet('background:black;')

        


        #웨이브폼 출력 영역
        #plt.rcParams["figure.figsize"] = (12,2)
        self.wavfig = plt.figure()
        # self.wavfig = plt.figure()
        self.c1_WaveForm = FigureCanvas(self.wavfig)


        # self.wavfigTarget = plt.figure(figsize=(10,4))
        # self.c2_WaveFormTarget = FigureCanvas(self.wavfigTarget)
        
        

        #타임 바 출력 영역
        self.d1_currentTimeLabel = QLabel()
        self.d1_currentTimeLabel.setSizePolicy(QSizePolicy.Preferred,
                                            QSizePolicy.Maximum)
        #self.d1_currentTimeLabel.setFixedSize(QSize)


        self.d2_positionSlider = QSlider(Qt.Horizontal)
        self.d2_positionSlider.setRange(0, 0)
        self.d2_positionSlider.sliderMoved.connect(self.setPosition)

        
        self.d3_totalTimeLabel = QLabel()
        self.d3_totalTimeLabel.setSizePolicy(QSizePolicy.Preferred,
                                          QSizePolicy.Maximum)
        

        #재생, 싱크 및 기타 버튼 영역
        
        self.e1_playButton = QPushButton()
        self.e1_playButton.setEnabled(False)
        self.e1_playButton.setFixedHeight(24)
        self.e1_playButton.setIconSize(btnSize)
        self.e1_playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.e1_playButton.clicked.connect(self.play)

        self.e2_cropVideo = QPushButton("Time Select")
        self.e2_cropVideo.setEnabled(False)
        self.e2_cropVideo.clicked.connect(self.exec_timeSelect)

        self.e3_faceRecog = QPushButton("Face Recognition")
        self.e3_faceRecog.setEnabled(False)
        self.e3_faceRecog.clicked.connect(self.exec_faceRecognition)

        self.e4_syncButton = QPushButton("Lip Sync")
        self.e4_syncButton.setEnabled(False)
        self.e4_syncButton.clicked.connect(self.exec_Wav2lip)
        
        self.e5_manual = QPushButton("Help")
        self.e5_manual.clicked.connect(self.exec_Help)
        
        #self.e6_settings.clicked.connect(self.settings_open)

        #프로그램 상태 출력 영역
        self.f1_state = QStatusBar()
        self.f1_state.setFont(QFont("default",13))
        self.f1_state.showMessage("Wait ...")
        
        #middle layout
        a_layout = QGridLayout()
        a_layout.addWidget(self.a1_1_OpenVideoBtn, 0, 0)
        a_layout.addWidget(self.a1_2_VideoFileName, 0, 1)
        a_layout.addWidget(self.a2_1_OpenAudioBtn, 1, 0)
        a_layout.addWidget(self.a2_2_AudioFileName, 1, 1)
        
        b_layout = QHBoxLayout()
        b_layout.addWidget(b1_videoWidget)

        ############################################
        c_layout = QHBoxLayout()
        #c_layout.addStretch(1)
        c_layout.addWidget(self.c1_WaveForm)
        #c_layout.addStretch(1)
        #c2_layout = QHBoxLayout()
        #c2_layout.addWidget(self.c2_WaveFormTarget)
        
        #time bar
        d_layout = QHBoxLayout()
        d_layout.addWidget(self.d1_currentTimeLabel)
        d_layout.addWidget(self.d2_positionSlider)
        d_layout.addWidget(self.d3_totalTimeLabel)
        

        cd_layout = QVBoxLayout()
        cd_layout.addLayout(c_layout)
        #cd_layout.addLayout(c2_layout)
        cd_layout.addLayout(d_layout)
        ############################################

        #buttons
        e_layout = QGridLayout()
        e_layout.addWidget(self.e1_playButton, 0, 0)
        e_layout.addWidget(self.e2_cropVideo, 0, 1)
        e_layout.addWidget(self.e3_faceRecog, 0, 2)
        e_layout.addWidget(self.e4_syncButton, 0, 3)
        e_layout.addWidget(self.e5_manual, 0, 4)

        #processbar
        f_layout = QHBoxLayout()
        f_layout.addWidget(self.f1_state)
        
        #first layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(a_layout)
        main_layout.addLayout(b_layout)
        #main_layout.addLayout(c_layout)
        main_layout.addLayout(cd_layout)
        main_layout.addLayout(e_layout)
        main_layout.addLayout(f_layout)
        
      
        
        self.setLayout(main_layout)

        


    def exec_timeSelect(self):
        self.TimeSelectClass = TimeSelect(self.Video_filename, self.Audio_filename, self.f1_state)
        self.TimeSelectClass.showModal()
        self.Video_filename = self.TimeSelectClass.get_filename()
        print(self.Video_filename)
        if(self.Video_filename != None):
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.Video_filename)))
            self.Draw_Waveform()
            self.play()
            self.f1_state.showMessage("Time Select Done...")

    def exec_faceRecognition(self):
        self.FaceRecognitionClass = FaceRecognition(self.Video_filename, self.Audio_filename, self.f1_state)
        self.FaceRecognitionClass.showModal()
        self.f1_state.showMessage("Face Recognition Done & StartLip syncing....")

    def exec_Help(self):
        self.HelpClass = Help(self.Video_filename, self.Audio_filename, self.f1_state)
        self.HelpClass.showModal()

    
    def exec_Wav2lip(self):
        self.Wav2lipThreadExecutionClass = ThreadClass(self.Video_filename, self.Audio_filename)
        self.Wav2lipThreadExecutionClass.start()
        self.f1_state.showMessage("Start lip-syncing. This may take more than 30 minutes... Work will be created in this folder.")


    # Video file은 무조건 존재함.
    def Draw_Waveform(self):
        #extraction
        Videoclip = mp.VideoFileClip(self.Video_filename)
        Videoclip.audio.write_audiofile("./temp/original.wav")

        #read file
        sr, sig = wavfile.read("./temp/original.wav")

        # self.wavfig.clear()
        ax = self.wavfig.add_subplot(111)

        ax.plot(sig, color='dodgerblue', linewidth=0.1)
        ax.set_title("Waveform of Video")
        ax.axes.xaxis.set_visible(False) #x축 라벨 제거
        ax.axes.yaxis.set_visible(False) #y축 라벨 제거
        # plt.tight_layout(pad=0,rect=(0,0,2,2))
        plt.tight_layout(pad=0)
        
        self.c1_WaveForm.draw()

    def AudioOpen(self):
        self.Audio_filename, _ = QFileDialog.getOpenFileName(self, "Audio Selection",
                ".", "Audio Files (*.mp3 *.wav)")
        if self.Audio_filename != None:
            self.a2_2_AudioFileName.showMessage(self.Audio_filename)
            self.f1_state.showMessage("Audio Upload Successful")
            #self.Draw_Waveform_Target()
            if self.Video_filename != None:
                self.e4_syncButton.setEnabled(True)
                self.e3_faceRecog.setEnabled(True)
                self.e2_cropVideo.setEnabled(True)
                self.f1_state.showMessage("Please Select Action. (Time Select, Face Recognition, Lip Sync)")
    
    
    def VideoOpen(self):
        self.Video_filename, _ = QFileDialog.getOpenFileName(self, "Video Selection",
                ".", "Video Files (*.mp4 *.avi)")
        if self.Video_filename != None:
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(self.Video_filename)))
            self.e1_playButton.setEnabled(True)
            if self.Audio_filename != None:
                self.e4_syncButton.setEnabled(True)
                self.e3_faceRecog.setEnabled(True)
                self.e2_cropVideo.setEnabled(True)
                self.f1_state.showMessage("Please Select Action. (Time Select, Face Recognition, Lip Sync)")
            self.a1_2_VideoFileName.showMessage(self.Video_filename)
            self.f1_state.showMessage("Video Upload Successful")
            #self.Draw_Waveform()
            self.play()
            


    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.resize(self.width(), self.height()+1)
        else:
            self.mediaPlayer.play()
            self.resize(self.width(), self.height()-1)

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.e1_playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.e1_playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))
    def positionChanged(self, position):
        self.d2_positionSlider.setValue(position)
        self.position = position
        if position >= 0:
            self.d1_currentTimeLabel.setText(self.hhmmss(position))
            
    def durationChanged(self, duration):
        self.d2_positionSlider.setRange(0, duration)
        if duration >= 0:
            self.d3_totalTimeLabel.setText(self.hhmmss(duration))
            self.finalFrame = duration
    
    def hhmmss(self, ms):
        mss = (ms%1000) / 10
        s = (ms / 1000) % 60
        m = (ms / (1000 * 60)) % 60
        h = (ms / (1000 * 60 * 60)) % 24
        return ("%02d:%02d:%02d:%02d" % (h, m, s, mss))
    
    def handleError(self):
        self.e1_playButton.setEnabled(False)
        self.a1_2_VideoFileName.showMessage("Error: " + self.mediaPlayer.errorString())
    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    sys.exit(app.exec_())



    
