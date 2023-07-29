from pathlib import Path
import logging
import sys
import os
from datetime import date


def create_dir_from_project_root(config_file_dir_level_lists: list) -> str:
    """Function constructs a directory path by combining the output of get_project_root function and unpacking a list containing the directory levels specified in config file.

    Example:
        The config_file_dir_level_lists provided is ["dir1", "dir2"],
    the directory created would be <project_root absolute_path>/dir1/dir2.

    Args:
        config_file_dir_level_lists (list): List of directories, representing the subsequent levels of directory in folder construct.

    Raises:
        None

    Returns:
        string representing a constructed directory path
    """

    # Add current working directory to the first entry input list. This is to facilitate the unpacking of list elements when constructing directory path
    config_file_dir_level_lists.insert(0, os.getcwd())

    # Unpack list to construct full path
    constructed_path = os.path.join(*config_file_dir_level_lists)

    return constructed_path


def get_logger(log_dir: list, include_debug: bool = True):
    """Function that creates a logger that generates 3 separate logs, capturing various log states (info, error or debug) for other scripts to use.

    Args:
        log_dir (list): List representing the levels of directory to store the logs referencing from project root folder
        include_debug (bool, optional): If true, includes .debug logging in .debug log file as part of log generation. Defaults to True.

    Raises:
        None

    Returns:
        None
    """

    log_formatter = logging.Formatter(
        "%(asctime)s | %(name)s |  %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y %H:%M",
    )
    get_current_date = str(date.today().strftime("%d%m%Y"))

    log = logging.getLogger("BACKTEST_LOG")

    # comment this to suppress console output. Sends message to streams like console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    log.addHandler(stream_handler)

    # Create daily date directory to store 3 logs levels file
    log_dir.insert(0, get_current_date)

    daily_log_directory = create_dir_from_project_root(log_dir)

    os.makedirs(daily_log_directory, exist_ok=True)

    # INFO LOGGING
    construct_logfile_info_name = get_current_date + ".log"
    logfile_info_path = os.path.join(daily_log_directory, construct_logfile_info_name)
    file_handler_info = logging.FileHandler(logfile_info_path, mode="a")
    file_handler_info.setFormatter(log_formatter)
    file_handler_info.setLevel(logging.INFO)
    log.addHandler(file_handler_info)

    # ERROR LOGGING
    construct_logfile_error_name = get_current_date + ".err"
    logfile_error_path = os.path.join(daily_log_directory, construct_logfile_error_name)
    file_handler_error = logging.FileHandler(logfile_error_path, mode="a")
    file_handler_error.setFormatter(log_formatter)
    file_handler_error.setLevel(logging.ERROR)
    log.addHandler(file_handler_error)
    log.setLevel(logging.INFO)

    # DEBUG LOGGING if inclue_debug is true
    if include_debug:
        construct_logfile_debug_name = get_current_date + ".debug"
        logfile_debug_path = os.path.join(
            daily_log_directory, construct_logfile_debug_name
        )

        file_handler_debug = logging.FileHandler(logfile_debug_path, mode="a")
        file_handler_debug.setFormatter(log_formatter)
        file_handler_debug.setLevel(logging.DEBUG)
        log.addHandler(file_handler_debug)
        log.setLevel(logging.DEBUG)

    return log


def get_project_root() -> Path:
    """Function that extracts the project root directory in absolute path.

    Args:
        None

    Raises:
        None

    Returns:
        Absolute path of project root directory located 2 levels above.
    """
    return Path(__file__).parent.absolute()

def get_project_root() -> str:
    """Get the project root path. Finds the pyproject.toml file which is located in the kedro project root and returns the project root filepath.

    Raises:
        RuntimeError: Reaches the root directory and cannot find the pyproject.toml file.

    Returns:
        str: kedro project root path
    """
    current_path = Path.cwd()
    while current_path != Path("/"):
        if (current_path / "pyproject.toml").is_file():
            return current_path
        current_path = current_path.parent
    raise RuntimeError("Kedro project root not found.")
