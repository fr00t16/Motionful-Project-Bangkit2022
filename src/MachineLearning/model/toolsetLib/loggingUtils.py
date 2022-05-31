import logging as log
#this is a utility function for logging
def init_logger():
    log.basicConfig(filename='log.txt', filemode='w',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=log.INFO, format='%(asctime)-15s %(message)s')
    console = log.StreamHandler()
    console.setLevel(log.INFO)
    log.getLogger('').addHandler(console)