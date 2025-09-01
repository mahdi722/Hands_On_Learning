import logging
from logging.handlers import RotatingFileHandler
import os
import json

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)


def get_logger(name: str = "agent", json_logs: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    file_formatter = JsonFormatter()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "agent.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
