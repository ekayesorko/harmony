import sounddevice
from timeit import default_timer as timer
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from time import sleep
import pickle
import math
import wave

class Beat:
    def __init__(self, rhythm=4, tempo=60.0):
        self.rhythm = rhythm
        self.tempo = tempo
        self.beat_interval = 60.0/tempo

    def get_beat_fraction(self):
        if (self.rhythm == 4):
            return 4
        else:
            return 2

    def get_timings(self, duration, subdivision=False):  # duration in seconds
        if (subdivision):
            times = np.arange(0, duration, self.beat_interval / self.get_beat_fraction())
        else:
            times = np.arange(0, duration, self.beat_interval)
        return times


    def get_frames(self, duration, subdivision=False, hop_length=512, sample_rate=22050):
        times = self.get_timings(duration=duration,subdivision=subdivision)
        return librosa.core.time_to_frames(times, hop_length=hop_length, sr=sample_rate)

    def cout(self):
        print((self.rhythm, self.tempo))

class Track:
    def __init__(self, sample_rate=22050, arr=None, duration=0):
        self.sample_rate = sample_rate
        self.arr = arr
        self.duration = duration

    def set_arr(self, arr, sample_rate=22050, duration=0):
        self.arr = arr.ravel()
        self.sample_rate = sample_rate
        self.duration = duration

    def get_arr(self):
        print(self.arr)
        return self.arr

    def assemble(self, another_track):
        if(self.arr.size > another_track.arr.size):
            padding = np.zeros(self.arr.size - another_track.arr.size)
            arr = np.append(another_track.arr,padding)
            return arr + self.arr
        else:
            padding = np.zeros(another_track.arr.size - self.arr.size)
            arr = np.append(self.arr,padding)
            return arr + another_track.arr


    def play(self):
        sounddevice.play(self.arr, samplerate=self.sample_rate)
        sounddevice.wait()

class Vocal(Track):
    def __init__(self,beat,duration,arr, sample_rate=22050):
        Track.__init__(self,sample_rate=sample_rate, duration=duration, arr = arr)
        self.beat = beat

    def analyze2(self, hop_length=512):
        chroma_cqt = librosa.feature.chroma_cqt(y=self.arr, sr=self.sample_rate)
        beat_frame_index = self.beat.get_frames(duration=self.duration, hop_length=hop_length,
                                                subdivision=True, sample_rate=self.sample_rate)
        beat_chroma = librosa.util.sync(chroma_cqt, beat_frame_index, aggregate=np.median)
        return beat_chroma


class Melody:
    def __init__(self):
        self.scale = None  # will be set when fit is complete
        self.root_note = None
        self.notations = []

    def __make_scales(self):
        scale = []
        # tones_list = [[2,2,1,2,2,2],[2,1,2,2,1,2]]
        tones_list = [[2, 2, 1, 2, 2, 2], [2, 1, 2, 2, 1, 2]]
        for i, tones in enumerate(tones_list):  # major ar minor
            scale.append([])
            for root_note in range(12):
                scale[i].append([])
                x = 0
                scale[i][root_note].append(root_note)
                for interval in tones:
                    x = x + interval
                    key = (root_note + x) % 12
                    scale[i][root_note].append(key)
                scale[i][root_note].sort()
        return scale

    def fit(self, chroma):  # output from voice.analyze2() is chroma
        scales = self.__make_scales()
        nrow, ncol = chroma.shape
        max_score = 0.0
        for i in range(2):
            for j in range(12):
                scale = scales[i][j]
                score = 0.0
                for note in scale:
                    for n in range(ncol):
                        if (chroma[note][n] > 0.90):
                            score += 1
                        elif (chroma[note][n] > 0.80):
                            score += 0.5
                # print(str(i) + " " + str(j)+ " " + str(score))
                if (score > max_score):
                    max_score = score
                    ans_i, ans_j = (i, j)
        self.scale = scales[ans_i][ans_j]
        self.root_note = ans_j
        print(self.scale)

    def simple_transform(self, chroma):
        # this will create an array of length chroma.ncol consisting of notes from the fitted class
        nrow, ncol = chroma.shape
        for beat_index in range(ncol):
            maxim = -1.0
            for note in self.scale:
                if (chroma[note][beat_index]) > maxim:
                    maxim = chroma[note][beat_index]
                    beat_note = note
            self.notations.append(beat_note)

class NaivePiano(Track):

    def __init__(self , melody, beat):
        Track.__init__(self,sample_rate=22050, arr= None, duration= len(melody.notations))
        self.melody = melody
        self.beat = beat
        self.sounds = self.__load_sounds()
        self.arr = self.__gen_music2(6)

    def __load_sounds(self,sample_rate = 22050):
        note_names = ["C5", "C5s", "D5", "D5s", "E5", "F5", "F5s", "G5", "G5s", "A5", "A5s", "B5"]
        notes = []
        for note_name in note_names:
            note, _ = librosa.load('Sounds/PianoSamples/' + note_name + '.wav', sr=44100)
            notes.append(note)
        return notes

    def __gen_music(self):
        music = np.array([])
        for note in self.melody.notations:
            duration =  math.floor(self.beat.beat_interval / self.beat.get_beat_fraction() * self.sample_rate)
            music = np.r_[music, self.sounds[note][:duration]]
        return music.ravel()

    def __gen_music2(self, measure):
        music = np.array([])
        pattern = [4,4,1,1,2,2,2]
        #pattern = [4,4,4,4]
        m = 0
        index = 0
        while(m<measure):
            for inc in pattern:
                note = self.melody.notations[index]
                duration = math.floor(self.beat.beat_interval * inc * self.sample_rate / self.beat.get_beat_fraction())
                print(duration)
                music = np.r_[music, self.sounds[note][:duration]]
                index += inc
            m+=1
        return music.ravel()

class Store:
    def __init__(self, beat_rhythm, beat_tempo, beat_interval, arr, arr_duration, sample_rate):
        self.beat_rhythm = beat_rhythm
        self.beat_tempo = beat_tempo
        self.beat_interval = beat_interval
        self.arr = arr
        self.arr_duration = arr_duration
        self.sample_rate = sample_rate

def load_shits():
    file = open('/media/orko/Storage/Sound/song_store','rb')
    storage = pickle.load(file)
    file.close()
    beat = Beat(storage.beat_rhythm, storage.beat_tempo)
    vocal = Vocal(beat,storage.arr_duration,storage.arr,storage.sample_rate)
    return (beat,vocal)

beat, vocal = load_shits()
analyzed = vocal.analyze2()
print(analyzed.shape)
melody = Melody()
melody.fit(analyzed)
melody.simple_transform(analyzed)
piano = NaivePiano(melody,beat)
print(piano.arr.shape)
print(vocal.arr.shape)

concert = vocal.assemble(piano)
print(concert.shape)
file = open('/media/orko/Storage/Sound/concert','wb')
pickle.dump(concert,file)
file.close()

# file = wave.open('output.wav','wb')
# file.setnchannels(1)
# file.setsampwidth(2)
# file.setframerate(22050)
# file.writeframes(vocal.arr)
# file.close()
