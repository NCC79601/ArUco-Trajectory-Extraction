import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.logger import get_logger

def test_logger():
    logger = get_logger('test_logger')
    logger.info('This is an info message')
    logger.debug('This is a debug message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')

def main():
    test_logger()

if __name__ == '__main__':
    main()