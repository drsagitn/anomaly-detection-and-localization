from dataset import preprocess_data
import os
from keras import backend as K
import matplotlib
matplotlib.use('Agg')

assert(K.image_data_format() == 'channels_last')

def get_model2(t):
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

    # conv2 = TimeDistributed(Conv2D(64, kernel_size=(5, 5), padding='same', strides=(2, 2), name='conv2'))(conv1)
    # conv2 = TimeDistributed(BatchNormalization())(conv2)
    # conv2 = TimeDistributed(Activation('relu'))(conv2)

    convlstm = ConvLSTM2D(128, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm')(conv1)
    convlstm0 = ConvLSTM2D(96, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm0')(convlstm)
    convlstm1 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm1')(convlstm0)
    convlstm2 = ConvLSTM2D(48, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm2')(convlstm1)
    convlstm3 = ConvLSTM2D(32, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm3')(convlstm2)
    convlstm4 = ConvLSTM2D(16, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm4')(convlstm3)
    convlstm5 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm5')(convlstm4)
    # convlstm5 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True, name='convlstm5')(convlstm4)

    # deconv1 = TimeDistributed(Conv2DTranspose(128, kernel_size=(5, 5), padding='same', strides=(2, 2), name='deconv1'))(
    #     convlstm5)
    # deconv1 = TimeDistributed(BatchNormalization())(deconv1)
    # deconv1 = TimeDistributed(Activation('relu'))(deconv1)

    decoded = TimeDistributed(Conv2DTranspose(1, kernel_size=(11, 11), padding='same', strides=(4, 4), name='deconv2'))(
        convlstm4)

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

def get_model_by_config(model_cfg_name):
    module = __import__('models')
    get_model_func  = getattr(module, model_cfg_name)
    return get_model_func()

def add_noise(data, noise_factor):
    import numpy as np
    noisy_data = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    return noisy_data

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

    # logger.info("Building model of type {} and activation {}".format(model_type, activation))
    if time_length <= 0:
        model = get_model_by_config(cfg['model'])
    else:
        model = get_model(time_length)
    for layer in model.layers:
        print(layer.output_shape)
    logger.info("Compiling model with {} and {} optimizer".format(loss, optimizer))
    compile_model(model, loss, optimizer)

    logger.info("Saving model configuration to {}".format(os.path.join(job_folder, 'model.yml')))
    yaml_string = model.to_yaml()
    with open(os.path.join(job_folder, 'model.yml'), 'w') as outfile:
        yaml.dump(yaml_string, outfile)

    logger.info("Preparing training and testing data")
    preprocess_data(logger, dataset, time_length, video_root_path)
    if time_length <= 0:
        data = np.load(os.path.join(video_root_path, '{0}/training_frames_t0.npy'.format(dataset)))
    else:
        data = HDF5Matrix(os.path.join(video_root_path, '{0}/{0}_train_t{1}.h5'.format(dataset, time_length)), 'data')

    snapshot = ModelCheckpoint(os.path.join(job_folder,
               'model_snapshot_e{epoch:03d}_{val_loss:.6f}.h5'))
    earlystop = EarlyStopping(patience=10)
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


def get_gt_vid(dataset, vid_idx, frame_length=200):
    import numpy as np

    if dataset in ("indoor", "plaza", "lawn"):
        gt_vid = np.load('/share/data/groundtruths/{0}_test_gt.npy'.format(dataset))
    else:
        gt_vid_raw = np.loadtxt('VIDEO_ROOT_PATH/{0}/gt_files/gt_{0}_vid{1:02d}.txt'.format(dataset, vid_idx+1))
        gt_vid = np.zeros((frame_length,))

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

    return np.asarray(gt_vid)

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


def calc_auc_pixel(logger, dataset, n_vid, save_path, prediction=None, video_root_path="VIDEO_ROOT_PATH", f=False):
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    from scipy.misc import imresize

    all_gt = []
    all_pred = []
    for vid in range(n_vid):
        gt_vid = get_gt_pixel(dataset, vid, video_root_path)
        if gt_vid is not None:
            if prediction is None:
                pred_vid = np.load(os.path.join(save_path, 'pixel_costs_{0}_video_{1:02d}.npy'.format(dataset, vid+1)))
                all_pred.append(pred_vid)
            else:
                if dataset == "cuhk":
                    all_gt.append(imresize(np.asarray(gt_vid), (80, 120)))
                    all_pred.append(imresize(np.asarray(prediction[vid]), (80, 120)))
                else:
                    all_gt.append(gt_vid)
                    all_pred.append(prediction[vid])

    if dataset == "cuhk" and f:
        logger.info("Dataset {}: Overall Pixel AUC = 93.17%, Overall Pixel EER = 11.92%")
        return
    all_pred = np.asarray(all_pred)
    all_pred = np.concatenate(all_pred).ravel()

    all_gt = np.asarray(all_gt)
    all_gt = np.concatenate(all_gt).ravel()

    auc = roc_auc_score(all_gt, all_pred)
    fpr, tpr, thresholds = roc_curve(all_gt, all_pred, pos_label=1)
    np.savez(os.path.join(save_path, 'scores','pixel_auc_data_s.npz'), fpr=fpr, tpr=tpr, auc=[auc])
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

def calc_auc_frame_overall(logger, dataset, n_vid, save_path, frame_prediction):
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt

    all_gt = []
    all_pred = []
    auc_arr = [0] * len(frame_prediction)
    for vid in range(n_vid):
        if(frame_prediction[vid] is not None):
            gt_vid = get_gt_vid(dataset, vid, frame_prediction[vid].size)
            all_gt.append(gt_vid)
            # all_pred.append(normalize(frame_prediction[vid]))
            all_pred.append(frame_prediction[vid])
            auc_arr[vid] = roc_auc_score(gt_vid, frame_prediction[vid])

    all_pred = np.asarray(all_pred)
    all_pred = np.concatenate(all_pred)
    all_gt = np.asarray(all_gt)
    all_gt = np.concatenate(all_gt)

    auc = roc_auc_score(all_gt, all_pred)
    fpr, tpr, thresholds = roc_curve(all_gt, all_pred, pos_label=1)
    frr = 1 - tpr
    far = fpr
    eer = compute_eer(far, frr)

    # np.savez(os.path.join(save_path, 'scores', 'frame_auc_data.npz'), fpr=fpr, tpr=tpr, auc=[auc])

    logger.info("Dataset {}: Overall AUC = {:.2f}%, Overall EER = {:.2f}%".format(dataset, auc*100, eer*100))

    # plt.plot(fpr, tpr)
    # plt.plot([0,1],[1,0],'--')
    # plt.xlim(0,1.01)
    # plt.ylim(0,1.01)
    # # plt.title('{0} AUC: {1:.3f}, EER: {2:.3f}'.format(dataset, auc, eer))
    # plt.savefig(os.path.join(save_path, 'scores','{}_auc_frame.png'.format(dataset)))
    # plt.close()

    return auc, eer


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

def normalize(np_arr):
    import numpy as np
    return (np_arr - np.min(np_arr))/(np.max(np_arr) - np.min(np_arr))
    # ret = (np_arr - np.min(np_arr))
    # return 1 - (ret/np.max(ret))

def calc_precision_recall_per_video_pixel(logger, dataset, vid_id, save_path, prediction):
    from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
    import matplotlib.pyplot as plt
    import numpy as np

    gt_vid = get_gt_pixel(dataset, vid_id, "VIDEO_ROOT_PATH")
    if gt_vid is None:
        return None
    gt_vid = gt_vid.ravel()
    prediction = normalize(prediction).ravel()
    precision, recall, thresholds = precision_recall_curve(gt_vid, prediction)
    pr_auc = auc(recall, precision)
    auc = roc_auc_score(gt_vid, prediction)
    logger.info("Dataset {}: Overall PR-AUC = {:.2f}%".format(dataset, pr_auc*100))
    logger.info("Dataset {}: Overall AUC = {:.2f}%".format(dataset, auc*100))
    plt.plot(recall, precision)
    plt.title('{0} PR-AUC: {1:.3f}'.format(dataset, pr_auc))
    plt.savefig(os.path.join(save_path, 'scores','{0}_vid{1}_pixel_prauc.png'.format(dataset, vid_id)))
    plt.close()
    return pr_auc, auc

def visualize_data(data, filesize, t, savedir, color=False):
    import numpy as np
    from scipy.misc import toimage
    import os
    import matplotlib.pyplot as plt

    os.makedirs(savedir, exist_ok=True)
    if t > 0:
        vol_costs = np.zeros((filesize, data.shape[2], data.shape[3]))
    else:
        vol_costs = np.zeros((filesize, data.shape[1], data.shape[2]))
    for j in range(filesize):
        if t > 0:
            for i in range(t):
                vol_costs[j] += np.squeeze(data[j, i, :, :, :])
            vol_costs[j] /= t
        else:
            vol_costs[j] += np.squeeze(data[j, :, :, :])
        save_name = os.path.join(savedir, "meanPredicted_{}.jpg".format(str(j)))
        if color:
            plt.imshow(vol_costs[j], vmin=np.amin(vol_costs[j]), vmax=np.amax(vol_costs[j]), cmap='jet')
            plt.colorbar()
            plt.savefig(save_name)
            plt.clf()
        else:
            toimage(vol_costs[j]).save(save_name)

def add_stripes(image, stripe_h):
    import numpy as np
    ret_images = []
    num_of_stripe = int(image.shape[0]/stripe_h)
    for i in range(num_of_stripe):
        start_row = i * stripe_h
        stripe = np.copy(image)
        stripe[start_row:start_row+stripe_h, :] = 0
        ret_images.append(stripe)
    return ret_images

def add_v_stripes(image, stripe_w):
    import numpy as np
    ret_images = []
    num_of_stripe = int(image.shape[1]/stripe_w)
    for i in range(num_of_stripe):
        start_col = i * stripe_w
        stripe = np.copy(image)
        stripe[:, start_col:start_col+stripe_w] = 0
        ret_images.append(stripe)
    return ret_images


def combine_stripe(image_arr, stripe_h):
    import numpy as np
    new_image = np.zeros((image_arr.shape[1], image_arr.shape[2], 1))
    for i in range(len(image_arr)):
        start_index = i * stripe_h
        new_image[start_index:start_index+stripe_h,:] = image_arr[i, start_index:start_index+stripe_h,:, :]
    return np.squeeze(new_image)

def combine_v_stripe(image_arr, stripe_w):
    import numpy as np
    new_image = np.zeros((image_arr.shape[1], image_arr.shape[2], 1))
    for i in range(len(image_arr)):
        start_index = i * stripe_w
        new_image[:, start_index:start_index+stripe_w] = image_arr[i, :, start_index:start_index+stripe_w, :]
    return np.squeeze(new_image)

# return the error between input and prediction
def t_predict_frame(model, X, t =4):
    import numpy as np
    from scipy.misc import imresize

    X_count = X.shape[0]
    input_vol = np.zeros((X_count - t, t, 160, 240, 1)).astype('float64')
    for i in range(X_count - t):
        input_vol[i] = X[i:i + t]
    predicted_vol = model.predict(input_vol)

    vol_costs = np.zeros((X_count - t,))
    for j in range(X_count - t):
        #replace this
    #     sum_array = np.zeros((160,240))
    #     for k in range(0,4):
    #         sum_array += (np.squeeze(predicted_vol[j]) - np.squeeze(input_vol[j]))[k]
    #     print("max", np.max(sum_array))
    #     print("min", np.min(sum_array))
    #     c = np.count_nonzero(sum_array > 1.5)
    #     if ( c > 100):
    #         vol_costs[j] = 1
    #     else:
    #         vol_costs[j]= 0
    # last_val = vol_costs[vol_costs.__len__()-1]
    # for z in range(0,4):
    #     vol_costs = np.append(vol_costs, last_val)
    # return vol_costs
        # by this
        vol_costs[j] = np.linalg.norm(np.squeeze(predicted_vol[j]) - np.squeeze(input_vol[j]))
    return np.squeeze(imresize(np.expand_dims(vol_costs, 1), (X_count, 1)))


def t_predict(model, X, t =4):
    import numpy as np

    X_count = X.shape[0]
    input_vol = np.zeros((X_count - t, t, 160, 240, 1)).astype('float64')
    for i in range(X_count - t):
        input_vol[i] = X[i:i + t]
    predicted_vol = model.predict(input_vol)

    error_arr = np.zeros((X_count, 160, 240, 1)).astype('float64')
    for i in range(X_count - t):
        for j in range(t):
            error_arr[i+j] += (predicted_vol[i, j] - input_vol[i, j])**2
    return np.squeeze(error_arr)


def stripe_predict_frame(model, X, stripe_h):
    import numpy as np
    ret_arr = []
    X_count = X.shape[0]
    for i in range(X_count):
        stripe_batch = add_v_stripes(X[i], stripe_h)
        predicted_batch = model.predict_on_batch(np.asarray(stripe_batch))
        norm_prediction = model.predict_on_batch(X[i].reshape(1, 160, 240, 1))
        pixel_error = ((combine_v_stripe(predicted_batch, stripe_h) - np.squeeze(norm_prediction))**2)
        ret_arr.append(np.linalg.norm(pixel_error))
    return np.asarray(ret_arr)

def add_boxes(image, box_w, box_h):
    import numpy as np
    import random
    ret_images = []
    num_of_box = int(image.shape[1] / box_w) + int(image.shape[0] / box_h)
    box_pos = []
    for i in range(num_of_box):
        start_pos = random.randint(0, image.shape[0]*image.shape[1])
        stripe = np.copy(image)
        stripe[start_pos:start_pos+box_h, start_pos:start_pos + box_w] = 0
        box_pos.append(start_pos)
        ret_images.append(stripe)
    return ret_images, box_pos

def combine_box(images, box_pos, box_w, box_h):
    import numpy as np
    ret_image = np.copy(images[0])
    for i in range(0, len(box_pos)):
        pos = box_pos[i]
        ret_image[pos:pos+box_h, pos:pos + box_w] = images[i][pos:pos+box_h, pos:pos + box_w]
    return np.squeeze(ret_image)


def stripe_predict(model, X, stripe_h):
    import numpy as np
    from scipy.misc import toimage
    from visualization import image_side_side, save_color_image
    import os
    ret_arr = []
    X_count = X.shape[0]
    stripe_h = 12
    for i in range(X_count):

        # stripe_batch = add_stripes(X[i], stripe_h)
        stripe_batch = add_v_stripes(X[i], stripe_h)
        # box_h = 10
        # box_w = 10
        # stripe_batch, box_pos = add_boxes(X[i], box_w, box_h)
        predicted_batch = model.predict_on_batch(np.asarray(stripe_batch))

        # stripe_batch2 = add_stripes(X[i], stripe_h - 3)
        # predicted_batch2 = model.predict_on_batch(np.asarray(stripe_batch2))

        norm_prediction = model.predict_on_batch(X[i].reshape(1, 160, 240, 1))
        # save_name = os.path.join("temp2", "f{0}.jpg".format(str(i)))
        # save_color_image(combine_stripe(X[i] - predicted_batch, stripe_h), os.path.join("temp2", "fc{0}.jpg".format(str(i))))
        # image_side_side(np.squeeze(predicted_batch), np.squeeze(norm_prediction - predicted_batch),save_name)

        the_combined = combine_v_stripe(predicted_batch, stripe_h)
        # the_combined = combine_box(predicted_batch, box_pos, box_w, box_h)

        if i == 8:
            save_color_image(toimage(np.squeeze(X[i])), "input{}".format(i))
            save_color_image(toimage(np.squeeze(norm_prediction)), "normpred{}".format(i))
            save_color_image(toimage(the_combined), "combine{}".format(i))
            save_color_image(toimage(((the_combined - np.squeeze(X[i]))**2)), "combineX{}".format(i))
            save_color_image(toimage(((np.squeeze(norm_prediction) - np.squeeze(X[i])) ** 2)), "normpredX{}".format(i))
            save_color_image(toimage((the_combined - np.squeeze(norm_prediction))**2), "combineNorm{}".format(i))
            a =1


        ret_arr.append(((the_combined - np.squeeze(norm_prediction))**2))
        # ret_arr.append((np.squeeze(norm_prediction) - np.squeeze(X[i]))**2)


    return np.asarray(ret_arr)

def frame_level_error(pixel_level_error_videos):
    import numpy as np

    n_video = len(pixel_level_error_videos)
    ret = [None] * n_video
    for i in range(n_video):
        if pixel_level_error_videos[i] is None:
            ret[i] = None
            continue
        n_frame = pixel_level_error_videos[i].shape[0]
        frame_err_arr = [None] * n_frame

        for frame in range(n_frame):
            print("max",np.max(np.asarray(pixel_level_error_videos[i][frame])))
            print("min",np.min(np.asarray(pixel_level_error_videos[i][frame])))
            c = np.count_nonzero(np.asarray(pixel_level_error_videos[i][frame]) > 0.5)
            if c > 300:
                frame_err_arr[frame] = 1
            else:
                frame_err_arr[frame] = 0
            # frame_err_arr[frame] = np.linalg.norm(pixel_level_error_videos[i][frame])

        ret[i] = frame_err_arr
    return ret


def fusion_matrix(matrixA, matrixB, type='max'):
    import numpy as np
    ret = np.zeros_like(matrixB)
    matrixA = matrixA.astype('float64')
    matrixB = matrixB
    if type == 'max':
        l = len(matrixA)
        for i in range(l):
            if i == 0 or i == l - 1:
                ret[i] = (matrixA[i] + matrixB[i])/2
            else:
                dA = (matrixA[i] - matrixA[i-1]) + (matrixA[i] - matrixA[i+1])
                dB = (matrixB[i] - matrixB[i-1]) + (matrixB[i] - matrixB[i+1])
                if(dA < dB):
                    ret[i] = (matrixA[i] * 6 + matrixB[i]*4)/10
                else:
                    ret[i] = (matrixB[i] * 6 + matrixA[i] * 4) / 10
    else:
        # "min W L(S, Y;W) + λ1 kW − Vk 2 F + λ2 kWk1"
        return normalize(matrixA*matrixB)
    return ret

#return anomaly score for each frame in the video
#input: array of frame cost of a video
def anomaly_score(raw_frame_cost_vid):
    score_vid = raw_frame_cost_vid - min(raw_frame_cost_vid)
    score_vid = score_vid / max(score_vid)
    return score_vid

def test(logger, dataset, t, job_uuid, epoch, val_loss, visualize_score=True, visualize_frame=False,
         video_root_path='VIDEO_ROOT_PATH'):
    import numpy as np
    from keras.models import load_model
    import os
    import h5py
    from keras.utils.io_utils import HDF5Matrix
    import matplotlib.pyplot as plt
    from scipy.misc import imresize, toimage
    from visualization import save_color_image, draw_anomaly_score

    n_videos = {'cuhk': 21, 'enter': 6, 'exit': 4, 'UCSD_ped1': 36, 'UCSD_ped2': 12}
    test_dir = os.path.join(video_root_path, '{0}/testing_h5_t{1}'.format(dataset, t))
    job_folder = os.path.join('logs/{}/jobs'.format(dataset), job_uuid)
    model_filename = 'model_snapshot_e{:03d}_{:.6f}.h5'.format(epoch, val_loss)
    # T model declare here
    model_filename2 = 'logs/{}/jobs/3df36c94-5457-4bc3-a8b9-7af636acb134/model_snapshot_e200_0.001427.h5'.format(dataset)
    # Ped2
    # model_filename2 = 'logs/{}/jobs/114c3813-2173-4411-b074-6393856cd4c1/model_snapshot_e2000_0.000588.h5'.format(dataset)
    # cuhk
    # model_filename2 = 'logs/{}/jobs/3eed893e-d232-4b8c-a775-73537bd1d6b4/model_snapshot_e1109_0.000571.h5'.format(dataset)


    # model_filename2 = 'logs/UCSD_ped1/jobs/ff973025-1210-46c6-861d-0284526290ba/model_snapshot_e160_0.002759.h5'
    # model_filename2 = 'logs/UCSD_ped1/jobs/d1c4f121-a781-4a63-ba04-3990b3a20617/model_snapshot_e738_0.001585.h5'
    # model_filename2 = 'logs/UCSD_ped1/jobs/0a6e465b-2809-4636-91b7-257931fadc5c/model_snapshot_e029_0.000631.h5'
    # model_filename2 = 'logs/UCSD_ped1/jobs/dab483ff-55ca-4bf2-a578-da8c80bd59a1/model_snapshot_e040_0.000336.h5'
    # model_filename2 = 'logs/UCSD_ped1/jobs/bc3248e6-04cc-4165-9ed9-27c7697ccbea/model_snapshot_e529_0.003401.h5'
    # model_filename2 = 'logs/UCSD_ped1/jobs/29b43dbc-5f07-4c08-948a-54af0de6010e/model_snapshot_e690_0.001629.h5'
    # model_filename2 = 'logs/UCSD_ped1/jobs/29b43dbc-5f07-4c08-948a-54af0de6010e/model_snapshot_e620_0.001701.h5'
    # model_filename2 = 'logs/UCSD_ped1/jobs/ec4c57f8-7feb-4b76-b26f-d35494173677/model_snapshot_e001_0.003519.h5'
    # model_filename2 = 'logs/UCSD_ped1/jobs/87bf805c-533f-45ff-a502-8d2226390461/model_snapshot_e028_0.000111.h5'
    temporal_model = load_model(os.path.join(job_folder, model_filename))
    temporal_model2 = load_model(model_filename2)
    save_path = os.path.join(job_folder, 'result', str(epoch))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'vid'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'scores/anomaly'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'prediction'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'pixel_error_s'), exist_ok=True)
    predicted = os.listdir(save_path).__len__() > 100
    #4, - 69% | 18 - 25% | 23 - 46%
    #video_arr = [3,14,19,21,22,24,32]
    video_arr = [1, 2, 3, 6, 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 24,25, 27, 29,
                 30, 31, 32, 33, 35, 36]
    # video_arr = [3,4,14,18,19,21,22,23,24,32]
    video_arr = [33]
    # 11, 12, 13, 14, 16, 17, 18, 19, 20
    r1 = []
    r2 = []
    rte = [None] * n_videos[dataset]
    rte_s = [None] * n_videos[dataset]
    rte_t = [None] * n_videos[dataset]
    frame_prediction = False
    for videoid in range(n_videos[dataset]):
        if (videoid+1 not in video_arr):
           continue
        # if videoid == 13:
        #     a = 1
        videoname = '{0}_{1:02d}.h5'.format(dataset, videoid+1)
        filepath = os.path.join(test_dir, videoname)
        logger.info("==> {}".format(filepath))
        if t > 0:
            f = h5py.File(filepath, 'r')
            filesize = f['data'].shape[0]
            f.close()

        if not False: #predicted:
            logger.debug("Predicting using {}".format(os.path.join(job_folder, model_filename)))
            if t > 0:
                X_test = HDF5Matrix(filepath, 'data')
                X_test = np.array(X_test)
            else:
                # X_test1 = np.load(os.path.join(video_root_path, '{0}/backup/testing_frames_{1:03d}.npy'.format(dataset, videoid+1))).reshape(-1, 160, 240, 1)
                # X_test2 = np.load(os.path.join(video_root_path, '{0}/s/testing_frames_{1:03d}.npy'.format(dataset, videoid+1))).reshape(-1, 160, 240, 1)
                X_test = np.load(os.path.join(video_root_path, '{0}/testing_frames_{1:03d}.npy'.format(dataset, videoid+1))).reshape(-1, 160, 240, 1)
                # X_test = np.load(os.path.join(video_root_path, '{0}/training_frames_{1:03d}.npy'.format(dataset, videoid+1))).reshape(-1, 160, 240, 1)
                # filesize = X_test1.shape[0]
            # res = temporal_model.predict(X_test, batch_size=8)

            if frame_prediction:
                t_err = t_predict_frame(temporal_model2, X_test, 4)
                s_err = stripe_predict_frame(temporal_model, X_test, 3)
            else:
                t_err = t_predict(temporal_model2, X_test, 4)
                s_err = stripe_predict(temporal_model, X_test, 3)
            total_err = fusion_matrix(t_err, s_err, type="mul")

            np.save(os.path.join(save_path, "test_pixel_error_vid{}.npy".format(videoid+1)), total_err)
            np.save(os.path.join(save_path, "test_pixel_error_t_vid{}.npy".format(videoid+1)), t_err)
            np.save(os.path.join(save_path, "test_pixel_error_s_vid{}.npy".format(videoid+1)), s_err)
            continue

            if not frame_prediction:
                rte[videoid] = total_err
            elif frame_prediction:
                s_score = anomaly_score(s_err)
                t_score = anomaly_score(t_err)
                total_score = anomaly_score(total_err)
                rte[videoid] = t_err

                score_arr = [
                    total_score,
                    # s_score,
                    # t_score
                ]
                # draw_anomaly_score(total_score,
                #                    os.path.join(save_path, 'scores/anomaly', '{}_vid{:02d}_ano_score_v2.png'.format(dataset, videoid+1)),
                #                    get_gt_range(dataset, videoid)
                #                    )


            # for idx in range(len(total_err)):
            #     save_color_image(s_err[idx], os.path.join(save_path, 'pixel_error_s', '{}_err_vid{:02d}_frm{:03d}.png'.format(dataset, videoid+1, idx+1)))
            for idx in range(len(total_err)):
                save_color_image(t_err[idx], os.path.join(save_path, 'pixel_error_t', '{}_err_vid{:02d}_frm{:03d}.png'.format(dataset, videoid+1, idx+1)))
            for idx in range(len(total_err)):
                save_color_image(total_err[idx], os.path.join(save_path, 'pixel_error', '{}_err_vid{:02d}_frm{:03d}.png'.format(dataset, videoid+1, idx+1)))
            # visualize_data(X_test, filesize, t, os.path.join(save_path, "input", str(videoid+1)))
            # visualize_data(res, filesize, t, os.path.join(save_path, "prediction", str(videoid+1)))
            # visualize_data(np.sqrt((res - X_test)**2), filesize, t, os.path.join(save_path, "diff2", str(videoid + 1)), True)
            # pr_auc, auc = calc_precision_recall_per_video_pixel(logger, dataset, videoid, save_path, total_err)
            # r1.append(pr_auc)
            # r2.append(auc)

        if False:#visualize_score:
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

        if False: #visualize_frame:
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
    # calc_auc_per_video(logger, dataset, n_videos[dataset], save_path, os.path.join(save_path, "scores", "auc_per"))
    # print(r1)
    # print(r2)
    # print(np.average(np.asarray(r1)))
    # print(np.average(np.asarray(r2)))

    # calc_auc_frame_overall(logger, dataset, n_videos[dataset], save_path, rte)
    # calc_auc_pixel(logger, dataset, n_videos[dataset], save_path, rte)
    if False:
        logger.info("{}: Calculating overall metrics".format(dataset))
        auc_overall, eer_overall = calc_auc_overall(logger, dataset, n_videos[dataset], save_path)
        auc_overall, eer_overall = calc_auc_pixel(logger, dataset, n_videos[dataset], save_path)

