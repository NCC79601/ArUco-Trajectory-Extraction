import os
import sys
import logging
from datetime import datetime

workdir = os.path.dirname(os.path.dirname(__file__))

class Logger:
    def __init__(self, name: str = 'default_logger'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())
        filename = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        filedir = os.path.join(workdir, 'logs')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        self.logger.addHandler(logging.FileHandler(os.path.join(
            filedir, filename
        )))
        self.logger.info(f'Logger {name} initialized')
    
    def info(self, msg):
        self.logger.info(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def critical(self, msg):
        self.logger.critical(msg)

_registered_loggers = {}

def register_logger(name: str):
    global _registered_loggers
    if name not in _registered_loggers:
        _registered_loggers[name] = Logger(name)
    return _registered_loggers[name]

def get_logger(name: str):
    global _registered_loggers
    if name not in _registered_loggers:
        return register_logger(name)
    return _registered_loggers[name]