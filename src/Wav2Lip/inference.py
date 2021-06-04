from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse

import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
from .models import Wav2Lip
from . import audio, face_detection
import platform

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

f1_state = None


#required -> 필수
###################################################################
parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=False) 
parser.add_argument('--face', type=str, 
					help='Filepath of video/image that contains faces to use', required=False)
parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source', required=False)

###################################################################
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 20, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()


def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('./temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()
#a_checkpoint_path=os.path.abspath("./checkpoints/wav2lip_gan.pth" #"
def main(a_video, a_audio, a_checkpoint_path = "./src/Wav2Lip/checkpoints/wav2lip_gan.pth"):
	args.face = a_video
	args.audio = a_audio
	args.checkpoint_path = a_checkpoint_path
	#args.outfile = os.path.expanduser('~') + "\\Desktop"
	args.outfile = "./result.avi"

	if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		args.static = True
	args.img_size = 96
  #args.face (=input_vid) 파일이 없을 경우
  # 0.1 Video to Frame
	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')
  
  # jpg, png, jpeg인 경우, full read 및 세팅
  # 0.2 Video to Frame
	elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(args.face)]
		fps = args.fps

  #video인 경우, full read 및 fps 세팅 # General case
  # 0.3 Video to Frame
	else:
		video_stream = cv2.VideoCapture(args.face) # video read
		fps = video_stream.get(cv2.CAP_PROP_FPS) #fps set
		
		print('Reading video frames...')

		full_frames = [] # defined full_frame array

    #모든 프레임에 설정 적용
    # 1. Frame adjustment
		while 1:
      #still_reading은 running state를 나타내는 bool 변수, frame은 실제 image
      #video_stream은 위에서 받은 cv2.VideoCapture object
			still_reading, frame = video_stream.read() 

      #break point
			if not still_reading: #비디오를 읽는중이 아니라면,
				video_stream.release() # open된 비디오 파일 혹은 영상장치를 닫음
				break
      #resize_factor는 해상도를 줄일 때 사용, default = 1임.
      #1보다 큰 값이 들어오는 경우는 480, 720p 와 같은 pixel 단위의 값을 입력하고, 이를 기준으로 해상도를 조절함.
      #원 연구자가 권장하는 값은 480, 720임.
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

      #rotate는 bool 변수이고, 휴대전화에서 찍은 영상이나 사진을 90' 회전하는데 사용
			if args.rotate:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

      #비디오를 더 작은 영역으로 자를 때 사용,
      #  default=[0, -1, 0, -1] 의 값을 갖고, -1을 기준점으로 비율로 계산.
      # 여러 인물이 등장하거나, 위에서 rotate를 실행시켜 화면을 조정하고 싶을 때 사용
			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

      #위에서 조절한 size로 frame 보관
			frame = frame[y1:y2, x1:x2]

      # 프레임 저장
			full_frames.append(frame)

	print ("Number of frames available for inference: "+str(len(full_frames)))

	# audio라는 arg에 받은 파일이 .wav로 끝나지 않은 경우
	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

    #subprocess로 command를 동작시킴.
    # 20210321 외부에서 가져오는 것인데, 이의 출처처를 모르겠음. ffmpeg에 비밀이 있을 것으로 생각
		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'
    
  #2. load audio
  # pre: audio.wav 파일과 sample rate를 입력받음
  # post: 16000의 sample rate 및 [-1 ~ 1]의 정규화된 범위를 가진 데이터로 저장
	wav = audio.load_wav(args.audio, 16000) #function call 
  
  #3. audio processing
  # filtering하고, scaling, normalization한 audio signal을 반환.
	mel = audio.melspectrogram(wav) #function call
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	batch_size = args.wav2lip_batch_size
  
  #4. data gen
	gen = datagen(full_frames.copy(), mel_chunks) #function call

	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
      #6. load model
			model = load_model(args.checkpoint_path) #function call
			print ("Model loaded")

			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter('./src/Wav2Lip/temp/result.avi', 
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
			print("only Video file save")
    #make tensor
		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()
	
	
	#command = 'C:/Users/Kang/Desktop/pytorch/Y2X/src/ffmpeg -y -i {} -i {} '.format(args.audio, 'C:/Users/Kang/Desktop/result.avi')
	#ffmpeg -i video.mp4 -i audio.mp4 result.mp4
	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, './src/Wav2Lip/temp/result.avi', args.outfile)
	subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	print(command)
	
if __name__ == '__main__':
	main()
