import tensorflow as tf
from .recorders import NVStatsRecorder, NVLinkStatsRecorder


class NVStats(tf.keras.callbacks.Callback):
    def __init__(self, gpu_index=0, interval=1, tensor_util=False, sudo_password=""):
        self.gpu_index = int(gpu_index)
        self.interval = float(interval)
        self.tensor_util = tensor_util
        self.sudo_password = str(sudo_password)

    def on_train_begin(self, logs=None):
        self.recorder = NVStatsRecorder(gpu_index=self.gpu_index, tensor_util=self.tensor_util, sudo_password=self.sudo_password)
        self.recorder.start(interval=self.interval)

    def on_train_end(self, logs=None):
        self.recorder.stop()
        self.data = self.recorder.get_data()
        self.throttle_reasons = self.data["throttle_reasons"]
        print("[NVStats] GPU Throttle:", self.throttle_reasons)


class NVLinkStats(tf.keras.callbacks.Callback):
    def __init__(self, sudo_password, gpus, interval=1.0, verbose=True):
        self.sudo_password = str(sudo_password)
        self.gpu_list = [int(i) for i in gpus]
        self.interval = float(interval)
        if verbose:
            print("[NVLinkStats] Watching:", self.gpu_list)

    def on_train_begin(self, logs=None):
        self.recorder = NVLinkStatsRecorder(self.sudo_password, self.gpu_list)
        self.recorder.start(interval=self.interval)

    def on_train_end(self, logs=None):
        self.recorder.stop()
        self.data = self.recorder.get_data()
