### data
raw_data_file_path = "../data/01_raw/stock_data.csv"
raw_data_tickers_file_path = "../data/01_raw/tickers.json"
raw_data_url = "https://www.advfn.com/nyse/newyorkstockexchange.asp?companies="
preprocessed_data_file_path = "data/02_preprocessed/new_preprocessed_data.csv"
# strategies
strategy_list = ["macd", "stock_trend"]
# model checkpoint filepath
tft_checkpoint_path = "best_tft.ckpt"
# --------------------------------------------------------------------------------------------
### model_specific_processing
scale_artefact_file_path = "data/04_model_specific_processing/artefacts/scaler.pkl"
scale_covariates_artefact_file_path = (
    "data/04_model_specific_processing/artefacts/scaler_covariates.pkl"
)
