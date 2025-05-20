from datetime import datetime
from enum import Enum
import os
from typing import List


class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class Logger:
    """로깅을 위한 클래스"""

    def __init__(self, log_to_file: bool = True):
        self.log_to_file = log_to_file
        self.logs: List[str] = []
        if log_to_file:
            os.makedirs("logs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = f"logs/workflow_{timestamp}.log"

    def log(self, message: str, level: LogLevel = LogLevel.INFO, keyword: str = None):
        """로그 메시지 기록"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"[{level.value}]"
        if keyword:
            prefix += f"[{keyword}]"

        log_message = f"{timestamp} {prefix} {message}"
        print(log_message)

        self.logs.append(log_message)
        if self.log_to_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_message + "\n")

    def info(self, message: str, keyword: str = None):
        self.log(message, LogLevel.INFO, keyword)

    def warning(self, message: str, keyword: str = None):
        self.log(message, LogLevel.WARNING, keyword)

    def error(self, message: str, keyword: str = None):
        self.log(message, LogLevel.ERROR, keyword)

    def success(self, message: str, keyword: str = None):
        self.log(message, LogLevel.SUCCESS, keyword)
