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
    
    def _run_command(self, cmd):
        return subprocess.getoutput(cmd)

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
        tensor_util (bool): Collect Tensor Core utilization
        sudo_password (str): Superuser password required to collect Tensor Core utilization
    Usage:
        ```python
        nv_stats_recorder = NVStatsRecorder(gpu_index=0)
        nv_stats_recorder.start(interval=1)
        # run your code here
        nv_stats_recorder.stop()
        gpu_data = nv_stats_recorder.get_data()
        ```
    """
    def __init__(self, gpu_index=0, tensor_util=False, sudo_password=""):
        self.stopped = False
        self.time_history = []
        self.sm_util_history = []
        self.tensor_util = tensor_util
        if self.tensor_util:
            print("\n[  IMPORTANT NOTICE                                                 ]")
            print("`tensor_util` is set to `True`")
            print("This enables Tensor Core utilization metrics collected via DCGM 1.7+")
            print("Supported GPUs are V100, T4 on Tesla-ready driver (418, 440+) only!")
            self.sudo_password = str(sudo_password)
            if len(self.sudo_password) < 2:
                print("This requires starting DCGM with superuser permissions! Please specify `sudo_password` argument.")
            output = self._start_dcgm_profiler()
            print("Attempted to start DCGM:", output.strip())
            print("[   END NOTICE                                                      ]\n")
        self.tensor_util_history = []
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
        self.gpu_index = str(gpu_index)
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
            "max_power": int(nvmlDeviceGetEnforcedPowerLimit(self.handle)),
            "temp_max": int(nvmlDeviceGetTemperatureThreshold(self.handle, 0)),
            "temp_slow": int(nvmlDeviceGetTemperatureThreshold(self.handle, 1)),
        }

    def _start_dcgm_profiler(self):
        return self._run_command(
            "echo '" + self.sudo_password + "' | sudo -S nv-hostengine"
        )
    
    def _get_tensor_pipe_active(self):
        try:
            output = self._run_command(
                "dcgmi dmon -e 1004 -c 1 -i " + self.gpu_index
            )
            output = float(output.split(" "+self.gpu_index+"  ")[1].strip()) * 100
            return output
        except Exception as e:
            print(e)
            print(output)
            return 0.0
        
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

    def _update(self, interval=2):
        t = 0
        while True:
            if self.stopped:
                nvmlShutdown()
                return
            else:
                st = time.time()
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
                self.pwr_history.append(nvmlDeviceGetPowerUsage(self.handle)/self.device_data["max_power"]*100)
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
                if self.tensor_util:
                    self.tensor_util_history.append(self._get_tensor_pipe_active())
                et = time.time()
                while (et-st) < interval:
                    time.sleep(interval/5)
                    et = time.time()
                t += (et-st)
                self.time_history.append(t)

    def plot_gpu_util(self, smooth=2, figsize=(10,6), dpi=100, outpath=None):
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
        pwr_history_history = self._moving_average(gpu_data["pwr_history"], smooth)[
            : t_len - trunc
        ]
        pcie_txrx = self._moving_average(gpu_data["pcie_txrx"], smooth)[: t_len - trunc]
        plt.clf()
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(time_history, sm_util_history, label="SM Util", color="g")
        if self.tensor_util:
            tensor_util_history = self._moving_average(gpu_data["tensor_util_history"], smooth)[
                : t_len - trunc
            ]
            plt.plot(time_history, tensor_util_history, label="TC Util", color="#76b900")
        plt.plot(time_history, mem_util_history, label="Mem I/O", color="c")
        plt.plot(time_history, gpu_data["mem_occupy_history"][trunc:t_len], label="Mem Usage", color="y")
        plt.plot(time_history, pwr_history_history, label="Power", color="m")
        plt.plot(time_history, pcie_txrx, label="PCIE Util", color="b")
        listed = []
        for item in gpu_data["throttle_reasons"]:
            label = str(item[0])
            if label not in listed:
                plt.axvline(item[1], color="r", linestyle="--", linewidth=1, label=label)
                listed.append(label)
            else:
                plt.axvline(item[1], color="r", linestyle="--", linewidth=1)
        plt.axhline(0, color="k", linewidth=1)
        plt.title("GPU Utilization")
        plt.ylabel("%")
        plt.xlabel("Time")
        plt.legend(loc="upper left")
        if outpath:
            plt.savefig(outpath)
        plt.show()
        
    def plot_gpu_temp(self, smooth=2, figsize=(10,6), dpi=100, outpath=None):
        gpu_data = self.get_data()
        trunc = smooth - 1
        t_len = len(gpu_data["time_history"])
        time_history = gpu_data["time_history"][trunc:t_len]
        temp_history = self._moving_average(gpu_data["temp_history"], smooth)[
            : t_len - trunc
        ]
        plt.clf()
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(time_history, temp_history, label="Temperature", color="b")
        listed = []
        for item in gpu_data["throttle_reasons"]:
            label = str(item[0])
            if label not in listed:
                plt.axvline(item[1], color="r", linestyle="--", linewidth=1, label=label)
                listed.append(label)
            else:
                plt.axvline(item[1], color="r", linestyle="--", linewidth=1)
        plt.axhline(self.device_data["temp_slow"], color="y", linestyle="--", label="Slowdown Temp")
        plt.axhline(self.device_data["temp_max"], color="r", linestyle="--", label="Max Temp")
        plt.axhline(0, color="k", linewidth=1)
        plt.title("GPU Temperature")
        plt.ylabel("degree C")
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
            "tensor_util_history": self.tensor_util_history[:t_len],
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
    
    def summary_seq(self, data_seq, idle_threshold=0.25):
        data_min = min(data_seq)
        data_max = max(data_seq)
        data_nonidle = []
        for d in data_seq:
            if d > idle_threshold:
                data_nonidle.append(d)
        steady_state = sum(data_nonidle)/len(data_nonidle)
        return (round(data_min, 1), round(steady_state, 1), round(data_max, 1))
    
    def summary(self):
        print("Summary (%)")
        print("===========")
        print("Metric  | Min    | Max    | Non-idle Avg")
        gpu_data = self.get_data()
        sm_summary = self.summary_seq(gpu_data["sm_util_history"])
        print("SM Util |", sm_summary[0], "|", sm_summary[2], "|", sm_summary[1])
        mem_summary = self.summary_seq(gpu_data["mem_util_history"])
        print("Mem I/O |", mem_summary[0], "|", mem_summary[2], "|", mem_summary[1])
        pwr_summary = self.summary_seq(gpu_data["pwr_history"])
        print("Power   |", pwr_summary[0], "|", pwr_summary[2], "|", pwr_summary[1])
        if self.tensor_util:
            tensor_summary = self.summary_seq(gpu_data["tensor_util_history"])
            print("Tensor  |", tensor_summary[0], "|", tensor_summary[2], "|", tensor_summary[1])


class NVLinkStatsRecorder(StatsRecorder):
    def __init__(self, sudo_password, gpus):
        self.stopped = False
        self.time_history = []
        self.gpus = [int(i) for i in gpus]
        self.nvlink_history = []
        self.sudo_password = str(sudo_password)
        self.colors = ["b", "g", "c", "m", "y", "b", "g", "c", "m", "y", "b", "g", "c", "m", "y", "b", "g", "c", "m", "y"]

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

    def _parse_nvlink_output(self, output, interval):
        """
        Returns the NVLink throughput across all GPUs
        """
        list_gpu_data = output.split("GPU ")[1:]
        gpu_bw = {}
        for i, gpu_data in enumerate(list_gpu_data):
            if i in self.gpus:
                list_tx, list_rx = [], []
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
                gpu_bw[str(i)] = total_bw / 1e6 / interval
        return gpu_bw

    def plot_nvlink_traffic(self, smooth=2, figsize=(10,6), dpi=100, outpath=None):
        nvlink_data = self.get_data()
        trunc = smooth - 1
        t_len = len(nvlink_data["time_history"])
        time_history = nvlink_data["time_history"][trunc:t_len]
        nvlink_history = nvlink_data["nvlink_history"]
        all_gpu_data = {}
        for gpu in self.gpus:
            per_gpu_data = []
            for data in nvlink_history:
                per_gpu_data.append(data[str(gpu)])
            per_gpu_data = np.asarray(per_gpu_data)
            per_gpu_data = self._moving_average(per_gpu_data, smooth)[:t_len - trunc]
            all_gpu_data[str(gpu)] = per_gpu_data
        plt.clf()
        plt.figure(figsize=figsize, dpi=dpi)
        for i, gpu in enumerate(self.gpus):
            color = self.colors[i]
            if i == 0:
                cum_gpu_data = all_gpu_data[str(gpu)]
                per_gpu_data = all_gpu_data[str(gpu)]
                plt.fill_between(time_history, per_gpu_data, np.zeros_like(per_gpu_data), color=color, alpha=0.5)
                plt.plot(time_history, per_gpu_data, color=color, label="GPU"+str(gpu))
            else:
                per_gpu_data = all_gpu_data[str(gpu)]
                plt.fill_between(time_history, per_gpu_data+cum_gpu_data, cum_gpu_data, color=color, alpha=0.5)
                cum_gpu_data += per_gpu_data
                plt.plot(time_history, cum_gpu_data, color=color, label="GPU"+str(gpu))
        plt.axhline(0, color="k", linewidth=1)
        plt.legend(loc="upper left")
        plt.title("NVLink Traffic")
        plt.ylabel("GB/s")
        plt.xlabel("Time")
        if outpath:
            plt.savefig(outpath)
        plt.show()

    def start(self, interval=2):
        Thread(target=self._update, args=([interval])).start()
        return self

    def _update(self, interval):
        t = 0
        counter = {}
        for i in self.gpus:
            counter[str(i)] = 0
        while True:
            if self.stopped:
                return
            else:
                st = time.time()
                self._reset_nvlink_counter()
                self.nvlink_history.append(counter)
                time.sleep(interval)
                output = self._read_nvlink_counter()
                counter = self._parse_nvlink_output(output, interval)
                et = time.time()
                t += (et-st)
                self.time_history.append(t)

    def get_data(self):
        data = {
            "time_history": self.time_history,
            "nvlink_history": self.nvlink_history,
        }
        return data
