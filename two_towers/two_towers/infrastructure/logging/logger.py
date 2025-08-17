import logging
import os
from logging import Logger
from typing import Optional


def setup_logging(noisy_libs: Optional[dict[str, int]] = None):
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    )
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", logging.INFO),
        handlers=[handler],
        force=True,
    )

    if noisy_libs is not None:
        for lib, level in noisy_libs.items():
            logging.getLogger(lib).setLevel(level)


class Logger:
    @staticmethod
    def get_logger(name: Optional[str] = None) -> Logger:
        return logging.getLogger(name)
