import setuptools

with open("pip_readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nvstatsrecorder",
    version="0.0.4",
    author="Timothy Liu",
    author_email="timothyl@nvidia.com",
    description="NVStatsRecorder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tlkh/NVStatsRecorder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
)
