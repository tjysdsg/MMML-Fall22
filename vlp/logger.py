import logging


def get_logger(log_path: str, logger_name: str):
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logger = logging.getLogger(logger_name)

    # print to file and console at the same time
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    logger.info(f"Logs are also saved at {log_path}")

    return logger
