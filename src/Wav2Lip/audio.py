import librosa
import librosa.filters
import numpy as np
# import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from .hparams import hparams as hp

def load_wav(path, sr):
    #sampling rate를 파라미터로 받으며, path에 있는 wav파일을 로드
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html  #ifilter prototype
        #IIR 혹은 FIR 필터를 사용하여 1차원을 따라 데이터 필터링
        #scipy.signal.lfilter(b, a, x, axis=- 1, zi=None
        #return The output of the digital filter.

        #[1, -k] : The numerator coefficient vector in a 1-D sequence.
        #[1]: The denominator coefficient vector in a 1-D sequence. If [1, -k] is not 1, then both a and b are normalized by [1, -k]
        # wav : An N-dimensional input array.
        return signal.lfilter([1, -k], [1], wav) # 필터링된 시그널 리턴
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav) 
    return wav

def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
        #hp.frame_shift_ms is (seconds)
        #1000 is (hop_size)
        #hp.sample_rate is (sample rate)
        # output length = (seconds) * (sample rate) / (hop size)
        # seconds만 바뀔 수 있는 변수임.
        # hop size = 1000, sr = 16000
    return hop_size

def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S

def melspectrogram(wav):
    #preemphasis -> wav를 filtering 한 signal 반환
    #D는 stft, Local weighted sums 을 STFT하고, transpose하여 반환
    #3.1 audio filtering
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize)) #function call (stft), function call (preemphasis)

    #수학적인 연산을 진행함
    #S는linear to mel 에서 어떤 처리를 한 signal을 넘기고, 이를 증폭
    #3.2 audio scaling
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db #function call (amp_to_db), function call (_linear_to_mel) # 20
    
    ############################################
    #3.3 audio normalization
    if hp.signal_normalization: # True
        return _normalize(S) #function call (normalize)
    return S

def _lws_processor():
    import lws
    # Does not work if fft_size is not multiple of hop_size!!
    # sample size = 20480, hop_size=256=12.5ms. fft_size는 window_size를 결정하는데, 2048을 시간으로 환산하면 2048/20480 = 0.1초=100ms
    # win_size 변수.
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech") #function call (get_hop_size)

#_stft는 use_lws의 셋팅에 따라 librosa를 사용하거나 lws를 사용
# LWS : Fast spectrogram phase recovery using Local Weighted Sums
# librosa : A python package for music and audio analysis.
def _stft(y):
    if hp.use_lws:
        #stft is Short Time Fourier Transform (STFT)
        return _lws_processor(hp).stft(y).T #function call (lws_processor)
    else:
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size) #function call (get_hop_size)

##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r
##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

# Conversions
_mel_basis = None

def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis() #mel basis를 받음 
    return np.dot(_mel_basis, spectogram) #mel basis와 abs(D)를 넘김
    #D는 wav의 Local weighted sums 을 STFT하고, transpose하여 반환 한 것.

def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2 
    # hp.fmax 보다 hp.sample_rate // 2 (= 8000)은 항상 크거나 같음
    #이를 mel 필터로 리턴
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)

def _amp_to_db(x):
    # amp
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S):
    #melspectogram을 사전 정의된 정규화 방법으로 정규화 할 경우,
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:#true
            #hyper parameter의 값을 바꾸지 않는 이상, 여기서 리턴함. -4, 4의 구간의 값으로 정규화
            #clip(array, min, max)
            # array = (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
            # min = -hp.max_abs_value = -4
            # max = hp.max_abs_value = 4
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)
    
    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))

def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value,
                              hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
    
    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
