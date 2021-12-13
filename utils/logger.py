import os, sys
import logging
from dotenv import load_dotenv
load_dotenv()

LOG_LEVEL = os.getenv('LOG_LEVEL')


def _get_logger(name):
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        f_handler = logging.StreamHandler(sys.stdout)
        f_handler.setLevel("INFO")
        format_str = '> %(asctime)s - %(levelname)s - %(name)s - %(message)s'
        f_format = logging.Formatter(format_str, "%d.%m. %H:%M")
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger

logger = _get_logger(__name__)
