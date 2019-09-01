import logging
import datetime
import os
import sys
import coloredlogs
from classifier import test

device = 'gpu0'
dataset = 'UCSD_ped1'
# dataset = 'cuhk'
job_uuid = '7cb9f8d8-a8df-48ff-90f3-6a385c74cb43'
job_uuid = '36b176dd-4a7f-4eba-aa95-dabf2145091c'
job_uuid = 'cd746ecf-d614-4e1a-a0a8-5a4e2d1275e1'
job_uuid = 'a14befd1-91dc-4a46-a034-5a3346b3874d'
job_uuid = 'ce1fd1d1-f170-4f1b-aa6a-4d85be5b4ce4'
job_uuid = '3df36c94-5457-4bc3-a8b9-7af636acb134'
job_uuid = 'ce1fd1d1-f170-4f1b-aa6a-4d85be5b4ce4'
job_uuid = '3df36c94-5457-4bc3-a8b9-7af636acb134'


# job_uuid = '8165b80e-14ed-407b-b5a4-9c3596f9abb8'
# epoch = 500
# val_loss = 0.001181
# time_length = 8

job_folder = os.path.join('logs/{}/jobs'.format(dataset), job_uuid)
log_path = os.path.join(job_folder, 'logs')
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_path, "test-{}.log".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))),
                    level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s")
coloredlogs.install()
logger = logging.getLogger()


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logger.warning("Ctrl + C triggered by user, testing ended prematurely")
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

# S model declare here
# UCSD Ped1
arr = [(10, 0.002689, 'db291bbf-55e9-40fd-b01f-43b4e474ec55', 0)]
arr = [(276, 0.001599, 'e08b85c0-fc44-47f2-b9e2-1bbac0006f88', 0)]
arr = [(86, 0.001149, '6ab962e0-b3bb-47d1-a513-4d68bdd2e08c', 0)]
arr = [(64, 0.000858, 'ae14934b-6961-45b1-b712-00381beef7dc', 0)]
arr = [(11, 0.003981, 'c0d6c0db-1a41-4089-9f05-ce9aaf7f9ccd', 0)]
arr = [(100, 0.003988, '968b2b8d-ba69-49fb-a27a-6d20787c4629', 0)]
arr = [(100, 0.003504, 'd1e9c23c-e754-4027-be93-111069ea3b81', 0)]
arr = [(298, 0.003159, 'f5846318-6990-4622-b584-ececf33a54d7', 0)]

# UCSD Ped2
# arr = [(749, 0.000554, '537f1112-8ec0-44eb-a380-c3a6e5a3c698', 0)]
# CUHK
# arr = [(227, 0.000455, 'c51f37de-9c2c-4941-b881-5029044a4564', 0)]

for item in arr:
    print("TESTING UUID", item[2])
    print("###Testing epoch", item[0])
    test(logger=logger, dataset=dataset, t=item[3], job_uuid=item[2], epoch=item[0], val_loss=item[1],
         visualize_score=True, visualize_frame=True)

# n_videos = {'avenue': 21, 'enter': 6, 'exit': 4, 'UCSD_ped1': 36, 'UCSD_ped2': 12}
# save_path = os.path.join(job_folder, 'result')
# auc_overall, eer_overall = calc_auc_overall(logger, dataset, n_videos[dataset], save_path)
# auc_overall, eer_overall = calc_auc_pixel(logger, dataset, n_videos[dataset], save_path)

logger.info("Job {} ({}) has finished testing.".format(job_uuid, dataset))
