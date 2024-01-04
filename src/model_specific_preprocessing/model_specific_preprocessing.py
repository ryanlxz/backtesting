import pandas as pd
import numpy as np
import os
from typing import Tuple, List, Union
from sklearn.preprocessing import StandardScaler
from darts import TimeSeries
import pickle
from src.feature_engineering.feature_engineering import create_month_and_quarter_feature
import conf
from conf import constants
from logs import logger


def run_model_specific_processing_pipeline() -> Tuple[List[TimeSeries], TimeSeries]:
    """_summary_

    Returns:
        Tuple[List[TimeSeries], TimeSeries]: Returns 3 lists of TimeSeries for train, val, test of stock prices, and 3 TimeSeries for covariates
    """
    raw_df_filepath = constants.preprocessed_data_file_path
    raw_df = pd.read_csv(raw_df_filepath, index_col=0)
    processed_df = process_raw_dataset(raw_df)
    covariates_df = create_month_and_quarter_feature(processed_df)
    # filter for only close prices from processed df as it also contains quarter and month columns
    processed_df = processed_df.filter(like="_Close")
    train_df, val_df, test_df = train_val_test_split(processed_df)
    covariates_train_df, covariates_val_df, covariates_test_df = train_val_test_split(
        covariates_df
    )
    logger.info("Start scaling and converting to timeseries")
    scaled_train_df, scaled_val_df, scaled_test_df = scale_data(
        train_df, val_df, test_df
    )

    # convert to timeseries for darts model
    train_series_list = convert_to_timeseries(scaled_train_df)
    val_series_list = convert_to_timeseries(scaled_val_df)
    test_series_list = convert_to_timeseries(scaled_test_df)
    cov_train_list = convert_to_timeseries(covariates_train_df, is_covariates=True)
    cov_val_list = convert_to_timeseries(covariates_val_df, is_covariates=True)
    cov_test_list = convert_to_timeseries(covariates_test_df, is_covariates=True)
    # ensure covariates are the same length as stock price (y-values)
    cov_train_list = [cov_train_list for _ in range(len(train_series_list))]
    cov_val_list = [cov_val_list for _ in range(len(val_series_list))]
    cov_test_list = [cov_test_list for _ in range(len(test_series_list))]
    logger.info("Completed scaling and converting to timeseries")
    return (
        train_series_list,
        val_series_list,
        test_series_list,
        cov_train_list,
        cov_val_list,
        cov_test_list,
    )


def process_raw_dataset(raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    """filters for Close price of stocks and removes any rows with missing values

    Args:
        raw_dataframe (pd.DataFrame): raw dataframe containing all price columns

    Returns:
        pd.DataFrame: processed dataframe which only contains the Close price and no missing values
    """
    rolling_mean_period = conf.backtest_conf["rolling_mean_period"]
    backfill_period = conf.backtest_conf["backward_fill_period"]
    processed_df = raw_dataframe.filter(like="_Close")
    processed_df = processed_df.fillna(
        processed_df.rolling(rolling_mean_period, min_periods=1, closed="left").mean()
    )
    processed_df = processed_df.bfill(limit=backfill_period)
    processed_df.dropna(axis=1, inplace=True)
    logger.info(
        f"Raw dataframe: {raw_dataframe.shape}, processed_dataframe: {processed_df.shape}"
    )
    return processed_df


def train_val_test_split(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame]:
    """Simple data split into train, val and test datasets.

    Args:
        dataframe (pd.DataFrame): dataset containing close price of stocks.

    Returns:
    Tuple containing the following:
    - train dataset (pd.DataFrame): training dataset
    - val dataset (pd.DataFrame): val dataset
    - test dataset (pd.DataFrame): test dataset
    """
    train_split = conf.backtest_conf["train_split_percentage"]
    val_split = conf.backtest_conf["val_split_percentage"]
    test_split = 1 - train_split - val_split
    if not 0 < train_split < 1 or not 0 <= val_split <= 1 or not 0 <= test_split <= 1:
        logger.debug(
            "Invalid value for train_split_percentage or val_split_percentage."
        )
    train_split = round(train_split * len(dataframe))
    val_split = round(val_split * len(dataframe)) + train_split
    train_df = dataframe.iloc[:train_split]
    val_df = dataframe.iloc[train_split:val_split]
    test_df = dataframe.iloc[val_split:]
    logger.info(
        f"Length of train_data: {len(train_df)},Length of val_data: {len(val_df)},Length of test_data: {len(test_df)}"
    )
    return train_df, val_df, test_df


# def scale_and_convert_to_timeseries(
#     train_df: pd.DataFrame,
#     val_df: pd.DataFrame,
#     test_df: pd.DataFrame,
#     is_covariates=False,
# ) -> List[TimeSeries]:
#     """_summary_

#     Args:
#         List (_type_): _description_

#     Returns:
#         List[TimeSeries]: list of TimeSeries
#     """
#     train_series_list = convert_to_timeseries(train_df, is_covariates)
#     val_series_list = convert_to_timeseries(val_df, is_covariates)
#     test_series_list = convert_to_timeseries(test_df, is_covariates)
#     train_scaled_list, val_scaled_list, test_scaled_list = list(), list(), list()

#     if is_covariates:
#         scaler_file_path = constants.scale_covariates_artefact_file_path
#         scale_fit(train_series_list, scaler_file_path)
#         train_series_scaled, val_series_scaled, test_series_scaled = scale_transform(
#             train_series_list, val_series_list, test_series_list, scaler_file_path
#         )
#         return train_series_scaled, val_series_scaled, test_series_scaled
# else:
#     scaler_file_path = constants.scale_artefact_file_path
#     scale_fit(train_series_list, scaler_file_path)
#     for train_series, val_series, test_series in zip(
#         train_series_list, val_series_list, test_series_list
#     ):
#         train_series_scaled, val_series_scaled, test_series_scaled = scale_data(
#             train_series, val_series, test_series
#         )
#         train_scaled_list.append(train_series_scaled)
#         val_scaled_list.append(val_series_scaled)
#         test_scaled_list.append(test_series_scaled)
# return train_scaled_list, val_scaled_list, test_scaled_list


def convert_to_timeseries(
    dataframe: pd.DataFrame, is_covariates=False
) -> List[TimeSeries]:
    """_summary_

    Args:
        dataframe (pd.DataFrame): _description_
        is_covariates (bool, optional): _description_. Defaults to False.

    Returns:
        List[TimeSeries]: _description_
    """
    dataframe.reset_index(drop=True, inplace=True)
    timeseries_list = [
        TimeSeries.from_series(dataframe.iloc[:, column])
        for column in range(dataframe.shape[1])
    ]
    if is_covariates:
        stacked_series = timeseries_list[0].stack(timeseries_list[1])
        for series in timeseries_list[2:]:
            stacked_series = stacked_series.stack(series)
        return stacked_series
    else:
        return timeseries_list


def scale_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> List[pd.DataFrame]:
    """Perform standardization on stock prices (y-values).
    First convert the stock price dataframes with multiple stocks (columns) into a single column for standardization across all stock prices.
    Performs 2 functions - scale_fit and scale_transform which fits Sklearn StandardScaler to the training data and transforms the data

    Args:
        train_df (pd.DataFrame): train df
        val_df (pd.DataFrame): val df
        test_df (pd.DataFrame): test df

    Returns:
        List[pd.DataFrame]: list of transformed dataframes
    """
    scaler_file_path = constants.scale_artefact_file_path
    pivot_df_list = []
    for i, df in enumerate([train_df, val_df, test_df]):
        melted_df = pd.melt(
            df, var_name="stock", value_name="Close", ignore_index=False
        )
        price_array = np.asarray(melted_df["Close"])
        price_array = price_array.reshape(-1, 1)
        # Perform fitting for the first iteration which is the training data
        if i == 0:
            scale_fit(price_array, scaler_file_path)
        scaled_array = scale_transform(price_array, scaler_file_path)
        scaled_df = pd.DataFrame(scaled_array, columns=["Close"])
        melted_df["Close"] = scaled_df["Close"].values
        pivot_df = melted_df.pivot(columns="stock", values="Close").reset_index()
        pivot_df = pivot_df.filter(like="_Close")
        pivot_df_list.append(pivot_df)
    return pivot_df_list


def scale_fit(train_series: pd.DataFrame, scaler_file_path: str) -> None:
    """Fit Sklearn StandardScaler object to training data, which converts values to a Z-score based on how many standard deviations the value is away from the mean.
    Save the fitted Scaler object as a pickle file.

    Args:
        train_series (TimeSeries): train timeseries

    Returns:
        True (bool)
    """
    dir_path = os.path.dirname(scaler_file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    scaler = StandardScaler()
    fitted_scaler = scaler.fit(train_series)
    with open(scaler_file_path, "wb") as file:
        pickle.dump(fitted_scaler, file)
    return True


def scale_transform(
    dataframe: Union[np.ndarray, pd.DataFrame],
    scaler_file_path: str,
) -> Union[np.ndarray, pd.DataFrame]:
    """transform an array or dataframe using the saved StandardScaler

    Args:
        dataframe (Union[np.ndarray, pd.DataFrame]): _description_
        scaler_file_path (str): _description_

    Returns:
        Union[np.ndarray, pd.DataFrame]: transformed array/dataframe
    """
    with open(scaler_file_path, "rb") as file:
        loaded_scaler = pickle.load(file)
    scaled_df = loaded_scaler.transform(dataframe)
    return scaled_df


if __name__ == "__main__":
    (
        train_scaled_list,
        val_scaled_list,
        test_scaled_list,
        cov_train_scaled_list,
        cov_val_scaled_list,
        cov_test_scaled_list,
    ) = run_model_specific_processing_pipeline()
