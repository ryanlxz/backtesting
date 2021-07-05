# NUS DSS template
a template repo for DSS (Data and Social Science) projects

You can modify this README.md to highlight:
1. Project overview
2. Methodology
3. Code examples (including notebooks)

# Project overview

Tell me something about this project.

# Methodology

Please refer to /docs

# Code, Notebook examples

Please refer to /notebooks

# Datasets

Please upload data to /data

Take note of privacy and copyrights.

# Statis resources

For storing static resources such as images, graphs etc

![](res/github_mark.png)

Please upload to /res

# Create new environment

install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
and create new virtual environment

create new environment, replace ```dss``` with the env name
```bash
conda create -n dss python==3.9
```

activate the new environment
```bash
conda activate dss
```

install essential packages
```bash
pip install -r requirements.txt
```


# Reproducing the codes

Please include requirements.txt
```python
conda list --export > requirements.txt
```

