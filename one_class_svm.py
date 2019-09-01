import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

dataset = "UCSD_ped1"
video_root_path = "VIDEO_ROOT_PATH"
n_videos = {'cuhk': 21, 'enter': 6, 'exit': 4, 'UCSD_ped1_train': 34, 'UCSD_ped1_test': 36, 'UCSD_ped2': 12}

def get_train_data(train_or_test, video_id=None):
    import os
    import numpy as np
    ret_data = None
    data_path = os.path.join(video_root_path, dataset, "svm", train_or_test)

    if video_id is not None:
        data_file = os.path.join(data_path, "{}_pixel_error_vid{}.npy".format(train_or_test, video_id))
        video_data = np.squeeze(np.load(data_file))
        ll = video_data.shape[0]
        video_data = video_data.reshape((ll, -1))
        return video_data
    for i in range(n_videos["{}_{}".format(dataset, train_or_test)]):
        data_file = os.path.join(data_path, "{}_pixel_error_vid{}.npy".format(train_or_test, i+1))
        video_data = np.squeeze(np.load(data_file))
        ll = video_data.shape[0]
        video_data = video_data.reshape((ll, -1))
        if ret_data is None:
            ret_data = video_data
        else:
            ret_data = np.concatenate((ret_data, video_data))
    return ret_data

def thres_distance_calculation(frame, thres):
    ret = 0
    for i in range(frame.shape[0]):
        if frame[i] > thres:
            ret = ret + i
    return ret

def reduce_to_2_dimension(data, thres):
    ret = []
    for k in range(data.shape[0]):
        # feature1 = np.count_nonzero(data[k] > thres)
        feature1 = thres_distance_calculation(data[k], thres)
        feature2 = np.linalg.norm(data[k])
        ret.append(np.asarray([feature1, feature2]))
    return np.asarray(ret)

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

# xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate train data
# X = 0.3 * np.random.randn(100, 2)
# X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
# X = 0.3 * np.random.randn(20, 2)
# X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
# X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X_train = get_train_data("train")
thres = 0.0001
X_train = reduce_to_2_dimension(X_train, thres)
# fit the model

clf = svm.OneClassSVM(nu=0.0001, kernel="rbf", gamma='auto')
clf.fit(X_train)

out_prediction = np.asarray([])
out_gt = np.asarray([])
for i in range(n_videos[dataset+"_test"]):
    X_test = get_train_data("test", i + 1)
    X_test = reduce_to_2_dimension(X_test, thres)
    plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=40, edgecolors='k')
    # Yi = clf.predict(X_test)
    Yi = clf.decision_function(X_test)
    GTi = get_gt_vid(dataset, i)
    out_prediction = np.concatenate((out_prediction, Yi<0))
    out_gt = np.concatenate((out_gt, GTi))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(out_gt, out_prediction)
print("AUC:", auc)

# y_pred_train = clf.predict(X_train)
# y_pred_test = clf.predict(X_test)
# y_pred_outliers = clf.predict(X_outliers)
# n_error_train = y_pred_train[y_pred_train == -1].size
# n_error_test = y_pred_test[y_pred_test == -1].size
# n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
#
# # plot the line, the points, and the nearest vectors to the plane
# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
#
# plt.title("Novelty Detection")
# plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
# a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
# plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
#
s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k', alpha=0.1)
# b2 = plt.scatter(out_prediction[:, 0], out_prediction[:, 1], c='blueviolet', s=s,
#                  edgecolors='k')
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
#                 edgecolors='k')
# plt.axis('tight')
# plt.xlim((-5, 5))
# plt.ylim((-5, 5))
# plt.legend([a.collections[0], b1, b2, c],
#            ["learned frontier", "training observations",
#             "new regular observations", "new abnormal observations"],
#            loc="upper left",
#            prop=matplotlib.font_manager.FontProperties(size=11))
# plt.xlabel(
#     "error train: %d/200 ; errors novel regular: %d/40 ; "
#     "errors novel abnormal: %d/40"
#     % (n_error_train, n_error_test, n_error_outliers))
plt.show()