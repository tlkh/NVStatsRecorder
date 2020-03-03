# NVStatsRecorder

![GitHub last commit](https://img.shields.io/github/last-commit/tlkh/NVStatsRecorder.svg) ![GitHub](https://img.shields.io/github/license/tlkh/NVStatsRecorder.svg) ![](https://img.shields.io/github/repo-size/tlkh/NVStatsRecorder.svg)

Simple Python utilities to instrument NVIDIA GPUs, designed to work directly from within Jupyter Notebooks requiring minimal system utilities. This allows users to quickly judge if their code takes proper advantage of NVIDIA GPUs and diagnose performance bottlenecks. **Not an official NVIDIA product.**

**Features**

* Works entirely from within Python
* Record NVIDIA GPU metrics without additional system utilities
* Helper functions to plot graphs visualising GPU usage
* NVLink bandwidth measurements (**require elevated permissions**)
* Returns raw data, including system-provided reasons for performance throttling
* Works from within Docker (when used with compatible GPU runtime)

## Usage

### Installing

```shell
pip install nvstatsrecorder
```

### For TensorFlow 2.0 (Keras)

You can easily instrument your TF 2.0 (Keras) model training, if you are using `model.fit()`. Otherwise, view the next section if you are using custom training loops.

**Example**

```python
from nvstatsrecorder.callbacks import NVStats, NVLinkStats

nv_stats = NVStats(gpu_index=0, interval=1)
nvlink_stats = NVLinkStats(SUDO_PASSWORD, gpus=[0,1,2,3])

model.fit(train_dataset,epochs=2,
          callbacks=[nv_stats, nvlink_stats])

gpu_data = nv_stats.data
nvlink_data = nvlink_stats.data

# you can also access the internal recorder object

nv_stats_recorder = nv_stats.recorder
nvlink_stats_recorder = nvlink_stats.recorder
```

### For any Python code

You can manually start and stop `NVStatsRecorder` or `NVLinkStatsRecorder` to instrument any Python code that uses GPU (e.g. PyTorch, MXNet, PyCUDA etc.)

**Example**

```python
from nvstatsrecorder.recorders import NVStatsRecorder, NVLinkStatsRecorder

# initialize recorders
nv_stats_recorder = NVStatsRecorder(gpu_index=0)
nvlink_stats_recorder = NVLinkStatsRecorder(SUDO_PASSWORD, gpus=[0,1,2,3])

# start recorders
nv_stats_recorder.start(interval=1)
nvlink_stats_recorder.start(interval=1)

# run your code here

# stop recorders
nv_stats_recorder.stop()
nvlink_stats_recorder.stop()

# get data from recorders
gpu_data = nv_stats_recorder.get_data()
nvlink_data = nvlink_stats_recorder.get_data()
```

### Plotting Graphs

```python
nv_stats_recorder.plot_gpu_util(smooth=3)
nvlink_stats_recorder.plot_nvlink_traffic(smooth=3)
```

<img src="https://raw.githubusercontent.com/tlkh/NVStatsRecorder/master/assets/nvstats.png" width="49%"> <img src="https://raw.githubusercontent.com/tlkh/NVStatsRecorder/master/assets/nvlinkstats.png" width="49%">

# Development

```shell
# build pip wheel and install
python3 setup.py sdist bdist_wheel
pip install dist/nvstatsrecorder-*-py3-none-any.whl
```

## Support

* Core Maintainer: [Timothy Liu (tlkh)](https://github.com/tlkh)
* **This is not an official NVIDIA product!**
* The website, its software and all content found on it are provided on an “as is” and “as available” basis. NVIDIA/NVAITC does not give any warranties, whether express or implied, as to the suitability or usability of the website, its software or any of its content. NVIDIA/NVAITC will not be liable for any loss, whether such loss is direct, indirect, special or consequential, suffered by any party as a result of their use of the libraries or content. Any usage of the libraries is done at the user’s own risk and the user will be solely responsible for any damage to any computer system or loss of data that results from such activities.
* Please open an issue if you encounter problems or have a feature request
