from keras.callbacks import Callback
import numpy as np
import os
import csv

class LossHistory(Callback):
    def __init__(self, job_folder, logger):
        super(LossHistory, self).__init__()
        self.save_path = job_folder
        self.logger = logger

    def on_train_begin(self, logs={}):
        self.logger.debug("Training started!")
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.logger.debug("Training loss for epoch {} is {}".format(epoch+1, logs.get('loss')))
        self.logger.debug("Validation loss for epoch {} is {}".format(epoch+1, logs.get('val_loss')))
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        with open(os.path.join(self.save_path, 'train_losses.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([logs.get('loss')])
        with open(os.path.join(self.save_path, 'val_losses.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([logs.get('val_loss')])

    def on_train_end(self, logs={}):
        self.logger.info('Saving training and validation loss history to file...')
        np.save(os.path.join(self.save_path, 'train_losses.npy'), np.array(self.train_losses))
        np.save(os.path.join(self.save_path, 'val_losses.npy'), np.array(self.val_losses))
