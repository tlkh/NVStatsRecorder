import time
from threading import Thread
import subprocess
import matplotlib.pyplot as plt
from pynvml import *
import numpy as np


class StatsRecorder(object):
    def __init__(self):
        self.stopped = False
        self.time_history = []

    def start(self, interval=1):
        Thread(target=self._update, args=([interval])).start()
        return self

    def _update(self, interval=1):
        t = 0
        while True:
            if self.stopped:
                return
            else:
                time.sleep(interval)
            self.time_history.append(t)
            t += interval

    def stop(self):
        self.stopped = True

    def _moving_average(self, a, n=2):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    def get_data(self):
        raise NotImplementedError


class NVStatsRecorder(StatsRecorder):
    """
    Used to record generic NVIDIA GPU metrics over a time period.
    Args:
        gpu_index (int): Index of GPU to instrument (starts from 0)
    Usage:
        ```python
        nv_stats_recorder = NVStatsRecorder(gpu_index=0)
        nv_stats_recorder.start(interval=1)
        # run your code here
        nv_stats_recorder.stop()
        gpu_data = nv_stats_recorder.get_data()
        ```
    """
    def __init__(self, gpu_index=0):
        self.stopped = False
        self.time_history = []
        self.sm_util_history = []
        self.sm_clocks_history = []
        self.mem_clocks_history = []
        self.mem_util_history = []
        self.mem_occupy_history = []
        self.temp_history = []
        self.pwr_history = []
        self.pcie_txrx = []
        self.throttle_reasons = []
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(gpu_index)
        # calculate max PCIE bandwidth
        try:
            pcie_gen = nvmlDeviceGetMaxPcieLinkGeneration(self.handle)
            pcie_gen_lane_bw = self._get_pcie_gen_lane_bw(pcie_gen)
            pcie_width = nvmlDeviceGetMaxPcieLinkWidth(self.handle)
            pcie_bandwidth = pcie_width * pcie_gen_lane_bw
        except Exception as e:
            print(e, "PCIE bandwidth information not available")
            pcie_bandwidth = -1
        # device throttle reasons
        self.throttle_reason_map = {
            "0": "None",
            "1": "GpuIdle",
            "2": "ApplicationsClocksSetting",
            "256": "DisplayClockSetting",
            "128": "HwPowerBrakeSlowdown",
            "8": "HwSlowdown",
            "64": "HwThermalSlowdown",
            "4": "SwPowerCap",
            "32": "SwThermalSlowdown",
            "16": "SyncBoost",
        }
        # device data dictionary
        self.device_data = {
            "device_name": str(nvmlDeviceGetName(self.handle)),
            "total_vram": int(nvmlDeviceGetMemoryInfo(self.handle).total / 1e9),
            "max_sm_clock": nvmlDeviceGetMaxClockInfo(self.handle, type=1),
            "max_mem_clock": nvmlDeviceGetMaxClockInfo(self.handle, type=2),
            "max_pcie_bandwidth": int(pcie_bandwidth),
        }

    def _get_throttle_reason(self, bitmask):
        bitmask = str(int(bitmask))
        return self.throttle_reason_map[bitmask]

    def _get_pcie_gen_lane_bw(self, gen):
        pcie_gen_bw = {
            "1": (250 * 1e3),
            "2": (500 * 1e3),
            "3": (985 * 1e3),
            "4": (1969 * 1e3),
        }
        return pcie_gen_bw[str(gen)]

    def _update(self, interval=1):
        t = 0
        while True:
            if self.stopped:
                nvmlShutdown()
                return
            else:
                util = nvmlDeviceGetUtilizationRates(self.handle)
                self.sm_util_history.append(util.gpu)
                self.mem_util_history.append(util.memory)
                self.sm_clocks_history.append(
                    nvmlDeviceGetClockInfo(self.handle, type=1)
                    / self.device_data["max_sm_clock"]
                    * 100
                )
                self.mem_clocks_history.append(
                    nvmlDeviceGetClockInfo(self.handle, type=2)
                    / self.device_data["max_mem_clock"]
                    * 100
                )
                memory_info = nvmlDeviceGetMemoryInfo(self.handle)
                self.mem_occupy_history.append(
                    memory_info.used / memory_info.total * 100
                )
                self.pwr_history.append(nvmlDeviceGetPowerUsage(self.handle) // 1000)
                self.temp_history.append(nvmlDeviceGetTemperature(self.handle, 0))
                try:
                    pcie_txrx = nvmlDeviceGetPcieThroughput(
                        self.handle, 0
                    ) + nvmlDeviceGetPcieThroughput(self.handle, 1)
                    self.pcie_txrx.append(
                        pcie_txrx / self.device_data["max_pcie_bandwidth"] * 100
                    )
                except:
                    self.pcie_txrx.append(0)
                throttle_reason = nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)
                throttle_reason_str = self._get_throttle_reason(throttle_reason)
                if (
                    throttle_reason_str not in self.throttle_reasons
                    and throttle_reason_str is not "None"
                ):
                    print("Detected GPU throttle:", throttle_reason_str)
                    self.throttle_reasons.append((throttle_reason_str, t))
                time.sleep(interval)
            self.time_history.append(t)
            t += interval

    def plot_gpu_util(self, smooth=2, show=True, figsize=(10,8), dpi=150, outpath=None):
        gpu_data = self.get_data()
        trunc = smooth - 1
        t_len = len(gpu_data["time_history"])
        time_history = gpu_data["time_history"][trunc:t_len]
        sm_util_history = self._moving_average(gpu_data["sm_util_history"], smooth)[
            : t_len - trunc
        ]
        mem_util_history = self._moving_average(gpu_data["mem_util_history"], smooth)[
            : t_len - trunc
        ]
        pcie_txrx = self._moving_average(gpu_data["pcie_txrx"], smooth)[: t_len - trunc]
        plt.clf()
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(time_history, sm_util_history, label="SM Util")
        plt.plot(time_history, mem_util_history, label="Mem I/O")
        plt.plot(time_history, gpu_data["mem_occupy_history"][trunc:t_len], label="Mem Usage")
        plt.plot(time_history, pcie_txrx, label="PCIE Util", color="y")
        for item in gpu_data["throttle_reasons"]:
            plt.axvline(item[1], color="r", linestyle="--", label=item[0])
        plt.title("GPU Utilization")
        plt.ylabel("%")
        plt.xlabel("Time")
        plt.legend(loc="upper left")
        if outpath:
            plt.savefig(outpath)
        plt.show()

    def get_data(self):
        t_len = len(self.time_history)
        data = {
            "device_data": self.device_data,
            "time_history": self.time_history[:t_len],
            "sm_util_history": self.sm_util_history[:t_len],
            "sm_clocks_history": self.sm_clocks_history[:t_len],
            "mem_util_history": self.mem_util_history[:t_len],
            "mem_clocks_history": self.mem_clocks_history[:t_len],
            "mem_occupy_history": self.mem_occupy_history[:t_len],
            "pcie_txrx": self.pcie_txrx[:t_len],
            "temp_history": self.temp_history[:t_len],
            "pwr_history": self.pwr_history[:t_len],
            "throttle_reasons": self.throttle_reasons,
        }
        return data


class NVLinkStatsRecorder(StatsRecorder):
    def __init__(self, sudo_password, gpus):
        self.stopped = False
        self.time_history = []
        self.gpus = [int(i) for i in gpus]
        self.nvlink_history = []
        self.sudo_password = str(sudo_password)
        self.counter = 0

    def _run_command(self, cmd):
        return subprocess.getoutput(cmd)

    def _start_nvlink_counter(self):
        return self._run_command(
            "echo '" + self.sudo_password + "' | sudo -S nvidia-smi nvlink -sc 0bz"
        )

    def _reset_nvlink_counter(self):
        return self._run_command(
            "echo '" + self.sudo_password + "' | sudo -S nvidia-smi nvlink -r 0"
        )

    def _read_nvlink_counter(self):
        return self._run_command(
            "echo '" + self.sudo_password + "' | sudo -S nvidia-smi nvlink -g 0"
        )

    def _parse_nvlink_output(self, output):
        """
        Returns the average NVLink throughput across all GPUs
        """
        list_gpu_data = output.split("GPU ")[1:]
        list_tx, list_rx = [], []
        for i, gpu_data in enumerate(list_gpu_data):
            if i in self.gpus:
                link_data = gpu_data.split("Link ")
                for link in link_data:
                    kbytes_list = link.split(" KBytes")
                    for kbyte in kbytes_list:
                        if "Tx" in kbyte:
                            tx = int(kbyte.split(": ")[-1])
                            list_tx.append(tx)
                        elif "Rx" in kbyte:
                            rx = int(kbyte.split(": ")[-1])
                            list_rx.append(rx)
        total_bw = sum(list_tx) + sum(list_rx)
        return total_bw / len(self.gpus)

    def plot_nvlink_traffic(self, smooth=2, show=True, figsize=(10,8), dpi=150, outpath=None):
        nvlink_data = self.get_data()
        trunc = smooth - 1
        t_len = len(nvlink_data["time_history"])
        time_history = nvlink_data["time_history"][trunc:t_len]
        nvlink_history = self._moving_average(nvlink_data["nvlink_history"], smooth)[
            : t_len - trunc
        ]
        plt.clf()
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(time_history, nvlink_history)
        plt.title("NVLink Traffic")
        plt.ylabel("GB/s")
        plt.xlabel("Time")
        if outpath:
            plt.savefig(outpath)
        plt.show()

    def start(self, interval=1):
        self._start_nvlink_counter()
        self._reset_nvlink_counter()
        Thread(target=self._update, args=([interval])).start()
        return self

    def _update(self, interval=1):
        t = 0
        while True:
            if self.stopped:
                return
            else:
                output = self._read_nvlink_counter()
                counter = self._parse_nvlink_output(output) / 1e6
                delta = counter - self.counter
                self.nvlink_history.append(delta / interval)
                self.counter = counter
                time.sleep(interval)
            self.time_history.append(t)
            t += interval

    def get_data(self):
        t_len = len(self.time_history)
        data = {
            "time_history": self.time_history[:t_len],
            "nvlink_history": self.nvlink_history[:t_len],
        }
        return data
