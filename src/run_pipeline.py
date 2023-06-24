from strate
import os
import sys

sys.path.append("../../")
import conf
from utils import get_project_root, get_logger, create_dir_from_project_root


if __name__ == "__main__":
    # Change working directory to project root
    os.chdir(get_project_root())

    # Call get_logger function to setup logger to be stored in subdirectory specified in a list of config file.
    logging = get_logger(
        log_dir=conf.backtest_conf["data_preprocessing"]["logging_dir"], include_debug=True
    )

    # Trigger series of preprocessing steps via function call
    start_preprocessing()

    logging.info("Exiting datapreprocessing program...")
