import logging

logging.basicConfig(
    format=("%(asctime)s | %(levelname)s | %(module)s"
            ":%(funcName)s:%(lineno)d | %(message)s"),
    level=logging.DEBUG,
)


def main():
    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    logging.critical("This is a critical message")


if __name__ == "__main__":
    main()
