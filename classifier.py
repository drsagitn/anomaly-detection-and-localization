from dataset import preprocess_data
import os
from keras import backend as K
import matplotlib
matplotlib.use('Agg')

assert(K.image_data_format() == 'channels_last')

def get_model2(t):
    from keras.models import Model
    from keras.layers.convolutional import Conv2D, Conv2DTranspose
    from keras.layers.normalization import BatchNormalization
    from keras.layers.core import Activation
    from keras.layers import Input

    input_tensor = Input(shape=(t, 160, 240, 1))

    conv1 = Conv2D(128, kernel_size=(7, 7), padding='same', strides=(4, 4), name='conv1')(input_tensor)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(64, kernel_size=(5, 5), padding='same', strides=(2, 2), name='conv2')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv3 = Conv2D(32, kernel_size=(3, 3), padding='same', strides=(1, 1), name='conv3')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)


    deconv1 = Conv2DTranspose(32, kernel_size=(3, 3), padding='same', strides=(1, 1), name='deconv1')(conv3)
    deconv1 = BatchNormalization()(deconv1)
    deconv1 = Activation('relu')(deconv1)

    deconv2 = Conv2DTranspose(64, kernel_size=(5, 5), padding='same', strides=(2, 2), name='deconv2')(deconv1)
    deconv2 = BatchNormalization()(deconv2)
    deconv2 = Activation('relu')(deconv2)

    deconv3 = Conv2DTranspose(128, kernel_size=(7, 7), padding='same', strides=(2, 2), name='deconv3')(deconv2)
    deconv3 = BatchNormalization()(deconv3)
    deconv3 = Activation('relu')(deconv3)

    decoded = Conv2DTranspose(1, kernel_size=(11, 11), padding='same', strides=(4, 4), name='deconv')(deconv3)

    return Model(inputs=input_tensor, outputs=decoded)

def get_model(t):
    from keras.models import Model
    from keras.layers.convolutional import Conv2D, Conv2DTranspose
    from keras.layers.convolutional_recurrent import ConvLSTM2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.wrappers import TimeDistributed
    from keras.layers.core import Activation
    from keras.layers import Input

    input_tensor = Input(shape=(t, 160, 240, 1))

    conv1 = TimeDistributed(Conv2D(128, kernel_size=(11, 11), padding='same', strides=(4, 4), name='conv1'),
                            input_shape=(t, 160, 240, 1))(input_tensor)
    conv1 = TimeDistributed(BatchNormalization())(conv1)
    conv1 = TimeDistributed(Activation('relu'))(conv1)

    conv2 = TimeDistributed(Conv2D(64, kernel_size=(5, 5), padding='same', strides=(2, 2), name='conv2'))(conv1)
    conv2 = TimeDistributed(BatchNormalization())(conv2)
    conv2 = TimeDistributed(Activation('relu'))(conv2)

    convlstm1 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm1')(conv2)
    convlstm2 = ConvLSTM2D(32, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm2')(convlstm1)
    convlstm3 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm3')(convlstm2)

    deconv1 = TimeDistributed(Conv2DTranspose(128, kernel_size=(5, 5), padding='same', strides=(2, 2), name='deconv1'))(convlstm3)
    deconv1 = TimeDistributed(BatchNormalization())(deconv1)
    deconv1 = TimeDistributed(Activation('relu'))(deconv1)

    decoded = TimeDistributed(Conv2DTranspose(1, kernel_size=(11, 11), padding='same', strides=(4, 4), name='deconv2'))(
        deconv1)

    return Model(inputs=input_tensor, outputs=decoded)


def compile_model(model, loss, optimizer):
    """Compiles the given model (from get_model) with given loss (from get_loss) and optimizer (from get_optimizer)
    """
    from keras import optimizers
    model.summary()

    if optimizer == 'sgd':
        opt = optimizers.SGD(nesterov=True)
    else:
        opt = optimizer

    model.compile(loss=loss, optimizer=opt)


def train(dataset, job_folder, logger, video_root_path='VIDEO_ROOT_PATH'):
    """Build and train the model
    """
    import yaml
    import numpy as np
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from custom_callback import LossHistory
    import matplotlib.pyplot as plt
    from keras.utils.io_utils import HDF5Matrix

    logger.debug("Loading configs from {}".format(os.path.join(job_folder, 'config.yml')))
    with open(os.path.join(job_folder, 'config.yml'), 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    nb_epoch = cfg['epochs']
    batch_size = cfg['batch_size']
    loss = cfg['cost']
    optimizer = cfg['optimizer']
    time_length = cfg['time_length']
    # shuffle = cfg['shuffle']

    # logger.info("Building model of type {} and activation {}".format(model_type, activation))
    model = get_model(time_length)
    logger.info("Compiling model with {} and {} optimizer".format(loss, optimizer))
    compile_model(model, loss, optimizer)

    logger.info("Saving model configuration to {}".format(os.path.join(job_folder, 'model.yml')))
    yaml_string = model.to_yaml()
    with open(os.path.join(job_folder, 'model.yml'), 'w') as outfile:
        yaml.dump(yaml_string, outfile)

    logger.info("Preparing training and testing data")
    preprocess_data(logger, dataset, time_length, video_root_path)
    data = HDF5Matrix(os.path.join(video_root_path, '{0}/{0}_train_t{1}.h5'.format(dataset, time_length)), 'data')

    snapshot = ModelCheckpoint(os.path.join(job_folder,
               'model_snapshot_e{epoch:03d}_{val_loss:.6f}.h5'))
    earlystop = EarlyStopping(patience=5)
    history_log = LossHistory(job_folder=job_folder, logger=logger)

    logger.info("Initializing training...")

    history = model.fit(
        data, data,
        batch_size=batch_size,
        epochs=nb_epoch,
        validation_split=0.15,
        shuffle='batch',
        callbacks=[snapshot, earlystop, history_log]
    )

    logger.info("Training completed!")
    np.save(os.path.join(job_folder, 'train_profile.npy'), history.history)

    n_epoch = len(history.history['loss'])
    logger.info("Plotting training profile for {} epochs".format(n_epoch))
    plt.plot(range(1, n_epoch+1),
             history.history['val_loss'],
             'g-',
             label='Val Loss')
    plt.plot(range(1, n_epoch+1),
             history.history['loss'],
             'g--',
             label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(job_folder, 'train_val_loss.png'))

def get_gt_range(dataset, vid_idx):
    import numpy as np
    ret =  np.loadtxt('VIDEO_ROOT_PATH/{0}/gt_files/gt_{0}_vid{1:02d}.txt'.format(dataset, vid_idx+1))
    if(ret.shape.__len__() == 1):
        return [ret]
    return ret


def get_gt_vid(dataset, vid_idx, pred_vid):
    import numpy as np

    if dataset in ("indoor", "plaza", "lawn"):
        gt_vid = np.load('/share/data/groundtruths/{0}_test_gt.npy'.format(dataset))
    else:
        gt_vid_raw = np.loadtxt('VIDEO_ROOT_PATH/{0}/gt_files/gt_{0}_vid{1:02d}.txt'.format(dataset, vid_idx+1))
        gt_vid = np.zeros_like(pred_vid)

        try:
            for event in range(gt_vid_raw.shape[0]):
                start = int(gt_vid_raw[event, 0]) - 1
                end = int(gt_vid_raw[event, 1])
                gt_vid[start:end] = 1
        except IndexError:
            start = int(gt_vid_raw[0])
            end = int(gt_vid_raw[1])
            gt_vid[start:end] = 1

    return gt_vid

def get_gt_pixel(dataset, vid_idx, video_root_path):
    from skimage.io import imread
    import os
    from skimage.transform import resize
    import numpy as np

    video_gt_dir = os.path.join(video_root_path, dataset, "gt", 'Test{0:03d}_gt'.format(vid_idx+1))
    if not os.path.isdir(video_gt_dir):
        return None
    gt_vid = []
    for file in sorted(os.listdir(video_gt_dir)):
        frame_value = imread(os.path.join(video_gt_dir, file), as_gray=True)/255
        frame_value = resize(frame_value, (160, 240), mode='reflect')
        gt_vid.append(np.round(frame_value))

    return gt_vid

def compute_eer(far, frr):
    cords = zip(far, frr)
    min_dist = 999999
    for item in cords:
        item_far, item_frr = item
        dist = abs(item_far-item_frr)
        if dist < min_dist:
            min_dist = dist
            eer = (item_far + item_frr) / 2
    return eer


def calc_auc_pixel(logger, dataset, n_vid, save_path, video_root_path="VIDEO_ROOT_PATH"):
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt

    all_gt = []
    all_pred = []
    for vid in range(n_vid):
        gt_vid = get_gt_pixel(dataset, vid, video_root_path)
        if gt_vid is not None:
            pred_vid = np.load(os.path.join(save_path, 'pixel_costs_{0}_video_{1:02d}.npy'.format(dataset, vid+1)))
            all_gt.append(gt_vid)
            all_pred.append(pred_vid)

    all_gt = np.asarray(all_gt)
    all_pred = np.asarray(all_pred)
    all_gt = np.concatenate(all_gt).ravel()
    all_pred = np.concatenate(all_pred).ravel()

    auc = roc_auc_score(all_gt, all_pred)
    fpr, tpr, thresholds = roc_curve(all_gt, all_pred, pos_label=1)
    frr = 1 - tpr
    far = fpr
    eer = compute_eer(far, frr)

    logger.info("Dataset {}: Overall Pixel AUC = {:.2f}%, Overall Pixel EER = {:.2f}%".format(dataset, auc*100, eer*100))

    plt.plot(fpr, tpr)
    plt.plot([0,1],[1,0],'--')
    plt.xlim(0,1.01)
    plt.ylim(0,1.01)
    plt.title('{0} AUC: {1:.3f}, EER: {2:.3f}'.format(dataset, auc, eer))
    plt.savefig(os.path.join(save_path, 'scores','{}_pixel_auc.png'.format(dataset)))
    plt.close()

    return auc, eer

def calc_auc_per_video(logger, dataset, n_vid, data_path, save_path):
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt

    os.makedirs(save_path, exist_ok=True)
    auc_arr = []
    for vid in range(n_vid):
        pred_vid = np.loadtxt(os.path.join(data_path, 'frame_costs_{0}_video_{1:02d}.txt'.format(dataset, vid + 1)))
        pred_vid = np.asarray(pred_vid).ravel()
        gt_vid = np.asarray(get_gt_vid(dataset, vid, pred_vid)).ravel()
        auc = roc_auc_score(gt_vid, pred_vid)
        auc_arr.append(auc)
        logger.info("{} video {}: Overall AUC = {:.2f}%".format(dataset, vid+1, auc * 100))
    ax = np.asarray(range(n_vid)) + 1
    plt.plot(ax, auc_arr)
    plt.xlim(0, n_vid)
    plt.ylim(0, 1.01)
    avg = np.sum(auc_arr) / n_vid
    plt.title('AUC across videos. AVG = {}'.format(avg))
    plt.savefig(os.path.join(save_path, '{}.png'.format(dataset)))
    plt.close()
    np.savetxt(os.path.join(save_path, '{}.txt'.format(dataset)), auc_arr)

def calc_auc_overall(logger, dataset, n_vid, save_path):
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt

    all_gt = []
    all_pred = []
    for vid in range(n_vid):
        pred_vid = np.loadtxt(os.path.join(save_path, 'frame_costs_{0}_video_{1:02d}.txt'.format(dataset, vid+1)))
        gt_vid = get_gt_vid(dataset, vid, pred_vid)
        all_gt.append(gt_vid)
        all_pred.append(pred_vid)

    all_gt = np.asarray(all_gt)
    all_pred = np.asarray(all_pred)
    all_gt = np.concatenate(all_gt).ravel()
    all_pred = np.concatenate(all_pred).ravel()

    auc = roc_auc_score(all_gt, all_pred)
    fpr, tpr, thresholds = roc_curve(all_gt, all_pred, pos_label=1)
    frr = 1 - tpr
    far = fpr
    eer = compute_eer(far, frr)

    logger.info("Dataset {}: Overall AUC = {:.2f}%, Overall EER = {:.2f}%".format(dataset, auc*100, eer*100))

    plt.plot(fpr, tpr)
    plt.plot([0,1],[1,0],'--')
    plt.xlim(0,1.01)
    plt.ylim(0,1.01)
    plt.title('{0} AUC: {1:.3f}, EER: {2:.3f}'.format(dataset, auc, eer))
    plt.savefig(os.path.join(save_path, 'scores','{}_auc.png'.format(dataset)))
    plt.close()

    return auc, eer

def visualize_data(data, filesize, t, savedir):
    import numpy as np
    from scipy.misc import toimage
    import os

    os.makedirs(savedir, exist_ok=True)
    vol_costs = np.zeros((filesize, data.shape[2], data.shape[3]))
    for j in range(filesize):
        for i in range(t):
            vol_costs[j] += np.squeeze(data[j, i, :, :, :])
        vol_costs[j] /= t
        toimage(vol_costs[j]).save(os.path.join(savedir, "meanPredicted_{}.jpg".format(str(j))))


def test(logger, dataset, t, job_uuid, epoch, val_loss, visualize_score=True, visualize_frame=False,
         video_root_path='VIDEO_ROOT_PATH'):
    import numpy as np
    from keras.models import load_model
    import os
    import h5py
    from keras.utils.io_utils import HDF5Matrix
    import matplotlib.pyplot as plt
    from scipy.misc import imresize

    n_videos = {'avenue': 21, 'enter': 6, 'exit': 4, 'UCSD_ped1': 36, 'UCSD_ped2': 12}
    test_dir = os.path.join(video_root_path, '{0}/testing_h5_t{1}'.format(dataset, t))
    job_folder = os.path.join('logs/{}/jobs'.format(dataset), job_uuid)
    model_filename = 'model_snapshot_e{:03d}_{:.6f}.h5'.format(epoch, val_loss)
    temporal_model = load_model(os.path.join(job_folder, model_filename))
    save_path = os.path.join(job_folder, 'result', str(epoch))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'vid'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'scores'), exist_ok=True)

    for videoid in range(n_videos[dataset]):
        videoname = '{0}_{1:02d}.h5'.format(dataset, videoid+1)
        filepath = os.path.join(test_dir, videoname)
        logger.info("==> {}".format(filepath))
        f = h5py.File(filepath, 'r')
        filesize = f['data'].shape[0]
        f.close()

        # gt_vid_raw = np.loadtxt('/share/data/groundtruths/gt_{0}_vid{1:02d}.txt'.format(dataset, videoid+1))

        logger.debug("Predicting using {}".format(os.path.join(job_folder, model_filename)))
        X_test = HDF5Matrix(filepath, 'data')
        res = temporal_model.predict(X_test, batch_size=4)
        X_test = np.array(X_test)

        if visualize_score:
            logger.debug("Calculating volume reconstruction error")
            vol_costs = np.zeros((filesize,))
            for j in range(filesize):
                vol_costs[j] = np.linalg.norm(np.squeeze(res[j])-np.squeeze(X_test[j]))

            file_name_prefix = 'vol_costs_{0}_video'.format(dataset)
            np.savetxt(os.path.join(save_path,file_name_prefix+'_'+'%02d'%(videoid+1)+'.txt'),vol_costs)

            logger.debug("Calculating frame reconstruction error")
            raw_costs = imresize(np.expand_dims(vol_costs,1), (filesize+t,1))
            raw_costs = np.squeeze(raw_costs)
            gt_vid = np.zeros_like(raw_costs)

            file_name_prefix = 'frame_costs_{0}_video'.format(dataset)
            np.savetxt(os.path.join(save_path, file_name_prefix+'_'+'%02d'%(videoid+1)+'.txt'), raw_costs)

            score_vid = raw_costs - min(raw_costs)
            score_vid = 1 - (score_vid / max(score_vid))

            file_name_prefix = 'frame_costs_scaled_{0}_video'.format(dataset)
            np.savetxt(os.path.join(save_path, file_name_prefix + '_' + '%02d' % (videoid + 1) + '.txt'), 1-score_vid)

            logger.debug("Plotting frame reconstruction error")
            plt.figure(figsize=(10, 3))
            plt.plot(np.arange(1, raw_costs.shape[0]+1), raw_costs)
            plt.savefig(os.path.join(save_path, '{}_video_{:02d}_err.png'.format(dataset, videoid+1)))
            plt.clf()

            logger.debug("Plotting regularity scores")
            plt.figure(figsize=(10, 3))
            ax = plt.subplot(111)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height*0.9])
            ax.plot(np.arange(1, score_vid.shape[0]+1), score_vid, color='b', linewidth=2.0)
            plt.xlabel('Frame number')
            plt.ylabel('Regularity score')
            plt.ylim(0, 1)
            plt.xlim(1, score_vid.shape[0]+1)

            vid_raw = get_gt_range(dataset, videoid)
            for event in vid_raw:
                plt.fill_between(np.arange(event[0], event[1]), 0, 1, facecolor='red', alpha=0.4)

            plt.savefig(os.path.join(save_path, 'scores','scores_{0}_video_{1:02d}.png'.format(dataset, videoid+1)), dpi=300)
            plt.close()

        if visualize_frame:
            logger.debug("Calculating pixel reconstruction error")
            count = 0
            for vol in range(filesize):
                for i in range(t):
                    pixel_costs[vol+i, :, :, :] += np.sqrt((res[count, i, :, :, :] - X_test[count, i, :, :, :])**2)
                count += 1

            file_name_prefix = 'pixel_costs_{0}_video'.format(dataset)
            np.save(os.path.join(save_path,file_name_prefix+'_'+'%02d'%(videoid+1)+'.npy'),pixel_costs)

            logger.debug("Drawing pixel reconstruction error")
            for idx in range(filesize+t):
                plt.imshow(np.squeeze(pixel_costs[idx]), vmin=np.amin(pixel_costs), vmax=np.amax(pixel_costs), cmap='jet')
                plt.colorbar()
                plt.savefig(os.path.join(save_path, 'vid', '{}_err_vid{:02d}_frm{:03d}.png'.format(dataset, videoid+1, idx+1)))
                plt.clf()

    logger.info("{}: Calculating overall metrics".format(dataset))
    auc_overall, eer_overall = calc_auc_overall(logger, dataset, n_videos[dataset], save_path)
    auc_overall, eer_overall = calc_auc_pixel(logger, dataset, n_videos[dataset], save_path)

