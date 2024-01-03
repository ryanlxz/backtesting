import yaml
import pathlib
import os
import datetime
import logging
import logging.config

path = pathlib.Path(__file__).parent / "logging_config.yaml"
with path.open(mode="rb") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    # Append the date stamp to the file name
    log_filename = config["handlers"]["file"]["filename"]
    base, extension = os.path.splitext(log_filename)
    today = datetime.datetime.today()
    log_filename = "{}{}{}".format(base, today.strftime("_%Y%m%d"), extension)
    config["handlers"]["file"]["filename"] = log_filename
    # Apply the configuration
    logging.config.dictConfig(config)
logger = logging.getLogger("staging")
