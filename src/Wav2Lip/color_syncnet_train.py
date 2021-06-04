from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet #2D covolution 연산을 하기 위한, 라이브러리 ./Wav2Lip/models/syncnet.py 에 정의
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)

args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    #해당 class가 시작하자마자, data_root에 있는 video를 frame으로 split하여, all_vidoes에 넣음.
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    #frame의 이름을 얻고, id로 반환
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    #start_id와, vid_name을 얻고, start id ~ start id + 5 까지, .jpg 형태로바꿔, 이름이 바뀐 프레임들을 리턴
    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T): # startid ~ startid + 5 까지 진행
            frame = join(vidname, '{}.jpg'.format(frame_id)) #.jpg형태로 이름 바꿈
            if not isfile(frame):
                return None
            window_fnames.append(frame) # window_frame에 추가
        return window_fnames #리턴

    #프레임의 시작을 fps만큼 자른 후, mel_step_size인 16만큼 더해서 한 단위로 만든 후 리턴
    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            # 0~frame의 길이만큼중에 랜덤한 값을 index로 만듬
            idx = random.randint(0, len(self.all_videos) - 1)
            # all_videos[idx]를 vidname에 대입
            vidname = self.all_videos[idx]

            #img name jpg파일 리스트를 img_names에 대입
            img_names = list(glob(join(vidname, '*.jpg'))) #vidname에 jpg를 합치고, glob 함
            
            #len이 15보다 작거나 같으면, 건너뜀
            if len(img_names) <= 3 * syncnet_T:
                continue
            # 15보다 클 때만 작동함.
            # 두 다른 이미지를 img_name과 wrong_img_name에 대입
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            #랜덤으로 y에 값을 채우고, chosen에 그에 맞는 img를 대입,
            if random.choice([True, False]):
                y = torch.ones(1).float() #스칼라 값 1로 채워진 텐서를 변수 인수의 사이즈 로 정의 된 모양 반환 torch.one()
                chosen = img_name
            else:
                y = torch.zeros(1).float() #스칼라 값 제로
                chosen = wrong_img_name

            #start_id와, vid_name을 얻고, chosen ~ Chosen + 5 까지, .jpg 형태로바꿔, 이름이 바뀐 프레임들을 리턴
            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname) #cv2를 이용해, 프레임을 읽음
                if img is None:
                    all_read = False
                    break
                try: #resize
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img) #window 변수에 추가

            # all read가 false면 건너뜀
            if not all_read: continue

            #wav 파일 조절 및 melspectogram으로 변환
            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y #tensor x와, mel chunck mel, y를 반환

logloss = nn.BCELoss()
#cosine loss 반환
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

#evaluation을 진행하고, model을 save
def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    
    while global_epoch < nepochs:
        running_loss = 0.
        #진행 바 표시 tadm
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            # cosine loss를 계산하기위한 필요 변수,
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            #cosine loss 계산
            loss = cosine_loss(a, v, y)

            #loss object, back propagation(역전파)를 통해 손실을 줄임.
            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            # checkpoint_interval=3000,
            if global_step == 1 or global_step % checkpoint_interval == 0:
                #3000번 마다 저장
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            #syncnet_eval_interval=10000,
            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    #횟수 10000번에, 출력하고 종료
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))

        global_epoch += 1

#void function, console에 loss를 출력
def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)

        return
# 저장
def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

#단순 로드
def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

#로드 및 epoch, step 설정
def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

# 만약 이 파일을 직접 실행한다면,
if __name__ == "__main__":
    # check point를 불러옴
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    # directory가 존재하지 않는다면, 만듦.
    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    # train_data를 loader함. data_utils.DataLoader은 torch 라이브러리에 있는 데이타 로드 함수임.
    # train_data_loader에는 모델을 업그레이드 시킬 수 있는, 훈련 데이터가 들어감.
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    # test_data_loader에는 업그레이드를 잘했지 평가할 수 있는 테스트 데이터가 들어감
    # DataLoader(dataset, batch_size = 64, processor의 개수 = 8 )
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8)

    # 빠른 훈련을 위해 torch에 device로 장치를 넘김
    device = torch.device("cuda" if use_cuda else "cpu")

    # Model을 들어온 프레임의 순서대로 2D convolution 연산 함. 이 때, device를 자원으로 사용.
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    #경사하강법 중, Adam이라는 방법을 사용하여 최적화 객체를 구성.
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)
    
    #checkpoint_path가 없다면,
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    #train을 호출하고 끝. 모든 정보를 train 함수로 넘김.
    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)
