class Beat:
    def __init__(self, rhythm=4, tempo=60.0):
        self.rhythm = rhythm
        self.tempo = tempo
        self.beat_interval = tempo / 60.0

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