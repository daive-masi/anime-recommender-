import logging

def get_logger(path):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    file = logging.FileHandler(path)
    logger.addHandler(file)
    return logger
