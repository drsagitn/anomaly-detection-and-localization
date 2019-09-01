def image_side_side(image_arr1, image_arr2, save_name):
    import matplotlib.pyplot as plt
    import numpy as np

    length = len(image_arr1)
    offset = 25
    if offset > length:
        offset = 0
    length = min(length, 25)
    fig, axarr = plt.subplots(length, 2)
    fig.set_size_inches(5, 50)
    for i in range(offset, length+offset):
        axarr[i-offset, 0].imshow(image_arr1[i], vmin=np.amin(image_arr1[i]), vmax=np.amax(image_arr1[i]), cmap='jet')
        axarr[i-offset, 1].imshow(image_arr2[i], vmin=np.amin(image_arr2[i]), vmax=np.amax(image_arr2[i]), cmap='jet')
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(save_name)
    plt.clf()

def save_color_image(image, save_name):
    import matplotlib.pyplot as plt
    import numpy as np
    ax = plt.subplot(111)
    plt.axis('off')
    im = ax.imshow(image, vmin=0, vmax=255, cmap='jet')
    plt.colorbar(im, ax=ax)
    plt.savefig(save_name)
    plt.clf()

def draw_anomaly_score(score_vid, save_name, gt_ranges=None):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 3))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    if(type(score_vid) is list):
        plt.xlim(1, score_vid[0].shape[0] + 1)
        custom_label = [
            "Fusion", "Appearance", "Temporal"
        ]
        for i in range(len(score_vid)):
            if i == 0: # fusion line
                ax.plot(np.arange(1, score_vid[i].shape[0] + 1), score_vid[i], label=custom_label[i], color='black', linewidth=2.2)
            elif i == 1: # appearance
                ax.plot(np.arange(1, score_vid[i].shape[0] + 1), score_vid[i], label=custom_label[i], color='forestgreen')
            else:
                ax.plot(np.arange(1, score_vid[i].shape[0] + 1), score_vid[i], label=custom_label[i], color='coral')
        plt.legend(loc=0)
    else:
        # import random
        # for i in range(700,710):
        #     score_vid[i] = random.uniform(0.35, 0.4)
        # for i in range(710,720):
        #     score_vid[i] = random.uniform(0.30, 0.35)
        # for i in range(720,740):
        #     score_vid[i] = random.uniform(0.25, 0.3)
        # for i in range(740,750):
        #     score_vid[i] = random.uniform(0, 0.2)
        ax.plot(np.arange(1, score_vid.shape[0] + 1), score_vid)#, color='b', linewidth=2.0)
        plt.xlim(1, score_vid.shape[0] + 1)
    plt.xlabel('Frame number')
    plt.ylabel('Anomaly score')
    plt.ylim(0, 1)



    if gt_ranges is not None:
        for event in gt_ranges:
            plt.fill_between(np.arange(event[0], event[1]), 0, 1, facecolor='red', alpha=0.4)

    plt.savefig(save_name, dpi=300)
    plt.close()


def load_auc_data_from_file(data_save_file):
    import numpy as np

    data = np.load(data_save_file)
    fpr = data['fpr']
    tpr = data['tpr']
    auc = data['auc']
    return fpr, tpr, auc


def draw_auc_graph_from_datafile(datafile_arr, save_path, custom_label_arr=None, custom_marker_arr=None):
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1., 0.1))
    ax.grid(linestyle=':')

    ll = len(datafile_arr)
    if custom_marker_arr is None:
        custom_marker_arr = [None] * ll
    if custom_label_arr is None:
        custom_label_arr = [""] * ll
    for i in range(ll):
        fpr, tpr, auc = load_auc_data_from_file(datafile_arr[i])
        if isinstance(auc[0], float):
            internal_lb = "{:0.1f}%".format(auc[0]*100)
        else:
            internal_lb = auc[0]
        if i == ll-1:
            plt.plot(fpr, tpr, label=custom_label_arr[i] + " " + internal_lb, marker=custom_marker_arr[i], color='black', linewidth=3)
        else:
            plt.plot(fpr, tpr, label=custom_label_arr[i] + " " + internal_lb, marker=custom_marker_arr[i])

    plt.plot([0, 1], [1, 0], '--', color='grey')
    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)

    plt.savefig(save_path)
    plt.close()

def calc_eer(auc_data_arr):
    from classifier import compute_eer
    for datafile in auc_data_arr:
        fpr, tpr, auc = load_auc_data_from_file(datafile)
        frr = 1 - tpr
        far = fpr
        eer = compute_eer(far, frr)
        print(eer, "\n")

def save_auc_manual_data(tpr, fpr, auc, file_name):
    import numpy
    numpy.savez(file_name, fpr=fpr, tpr=tpr, auc=[auc])

def create_auc_sample_data():
    import numpy as np
    auc = 0.77
    tpr = np.asarray([0, 0.26, 0.39, 0.6, 0.71, 0.8, 0.89, 1])
    fpr = np.asarray([0, 0.03, 0.07, 0.15, 0.24, 0.38, 0.69, 1])
    save_auc_manual_data(tpr, fpr, auc, "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_zhang.npz")

    auc = 0.441
    tpr = np.asarray([0, 0.03, 0.12, 0.19, 0.23, 0.29, 0.36, 0.4, 0.46, 0.68, 1])
    fpr = np.asarray([0, 0.05, 0.07, 0.1, 0.12, 0.19, 0.3, 0.42, 0.58, 0.9, 1])
    save_auc_manual_data(tpr, fpr, auc, "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_Mahadevan.npz")

    auc = 0.524  # used as label
    tpr = np.asarray([0, 0.03, 0.1, 0.19, 0.25, 0.4, 0.49, 0.58, 0.68, 0.8, 0.83, 1])
    fpr = np.asarray([0, 0.1, 0.19, 0.27, 0.3, 0.37, 0.44, 0.47, 0.65, 0.69, 0.73, 1])
    save_auc_manual_data(tpr, fpr, auc, "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_Tudo.npz")

    auc = 0.638  # used as label
    tpr = np.asarray([0, 0.26, 0.31, 0.39, 0.5, 0.58, 0.63, 0.72, 0.82, 0.93, 0.97, 1])
    fpr = np.asarray([0, 0.03, 0.09, 0.22, 0.33, 0.38, 0.46, 0.63, 0.7, 0.84, 0.88, 1])
    save_auc_manual_data(tpr, fpr, auc,
                         "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_Lu.npz")

    auc = 0.672  # used as label
    tpr = np.asarray([0, 0.06, 0.18, 0.26, 0.35, 0.42, 0.5, 0.59, 0.63, 0.69, 1])
    fpr = np.asarray([0, 0.01, 0.02, 0.04, 0.05, 0.11, 0.23, 0.4, 0.51, 0.59, 1])
    save_auc_manual_data(tpr, fpr, auc,
                         "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_Xu.npz")


# calc_eer(["logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data.npz",
#                               "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_t.npz",
#                               "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_s.npz"])
# draw_auc_graph_from_datafile([
#                               # "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_zhang.npz",
#                               # "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_Mahadevan.npz",
#                               # "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_Tudo.npz",
#                               # "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_Lu.npz",
#                               # "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_Xu.npz",
#                                 "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_t.npz",
#                                 "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_s_without_inpainting.npz",
#                                 "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_s.npz",
#                                 "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data_fusion_without_inpainting.npz",
#                                 "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_data.npz"
#                             ],
#                              "logs/UCSD_ped1/jobs/f5846318-6990-4622-b584-ececf33a54d7/result/298/scores/pixel_auc_compare_methodsv5.png",
#                              [
#                                  # "Zhang",
#                                  # "Mahadevan",
#                                  # "Tudo",
#                                  # "Lu",
#                                  # "Xu",
#                                  "Temporal",
#                                 "Appearance without inpainting",
#                                 "Appearance",
#                                  "Fusion without inpainting",
#                                 "Fusion"
#                              ],
#                              [
#                                  # "<",
#                                  # "o",
#                                  # ">",
#                                  # "v",
#                                  # "p",
#                                  None, None, None, None, None]
#                              )
# create_auc_sample_data()
