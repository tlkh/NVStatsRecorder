# NVStatsRecorder

Simple Python utilities to instrument NVIDIA GPUs, designed to work directly from within Jupyter Notebooks requiring minimal system utilities.

This allows users to quickly judge if their code takes proper advantage of NVIDIA GPUs and diagnose performance bottlenecks.

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

## Support

* Core Maintainer: [Timothy Liu (tlkh)](https://github.com/tlkh)
* **This is not an official NVIDIA product!**
* The website, its software and all content found on it are provided on an “as is” and “as available” basis. NVIDIA/NVAITC does not give any warranties, whether express or implied, as to the suitability or usability of the website, its software or any of its content. NVIDIA/NVAITC will not be liable for any loss, whether such loss is direct, indirect, special or consequential, suffered by any party as a result of their use of the libraries or content. Any usage of the libraries is done at the user’s own risk and the user will be solely responsible for any damage to any computer system or loss of data that results from such activities.
* Please open an issue if you encounter problems or have a feature request
