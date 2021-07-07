# Backtesting trading strategies
Simple to implement trading strategies like MACD indicator, moving average and candlestick patterns are backtested on a dataset consisting of the top 100 stocks in terms of market capitalization from the S&P500 index.   

# Methodology

Please refer to /docs

# Code, Notebook examples

Please refer to /notebooks


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

