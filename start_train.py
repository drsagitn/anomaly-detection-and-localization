import logging
import datetime
import os
import sys
import coloredlogs
from classifier import train
import uuid
from shutil import copyfile

dataset = 'UCSD_ped1'
device = 'gpu0'

job_uuid = str(uuid.uuid4())
job_folder = os.path.join('/home/thinh/anomaly/github/abnormal-spatiotemporal-ae/logs/{}/jobs'.format(dataset), job_uuid)
os.makedirs(job_folder)
copyfile('config.yml', os.path.join(job_folder, 'config.yml'))

log_path = os.path.join(job_folder, 'logs')
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_path,
                                          "train-{}.log".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))),
                    level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s")
coloredlogs.install(level=logging.INFO)
logger = logging.getLogger()


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logger.warning("Ctrl + C triggered by user, training ended prematurely")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

if device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    logger.debug("Using CPU only")
elif device == 'gpu0':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    logger.debug("Using GPU 0")
elif device == 'gpu1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    logger.debug("Using GPU 1")
elif device == 'gpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    logger.debug("Using GPU 0 and 1")


train(dataset=dataset, job_folder=job_folder, logger=logger)

logger.info("Job {} has finished training.".format(job_uuid))
