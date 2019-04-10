def calc_mean(dataset, video_root_path='/share/data/videos'):
    import os
    from skimage.io import imread
    import numpy as np
    from skimage.transform import resize

    frame_path = os.path.join(video_root_path, dataset, 'training_frames')
    count = 0
    frame_sum = None

    try:
        for frame_folder in os.listdir(frame_path):
            print('==> ' + os.path.join(frame_path, frame_folder))
            for frame_file in os.listdir(os.path.join(frame_path, frame_folder)):
                frame_filename = os.path.join(frame_path, frame_folder, frame_file)
                frame_value = imread(frame_filename, as_gray=True)/256
                frame_value = resize(frame_value, (160, 240), mode='reflect')
                assert(0. <= frame_value.all() <= 1.)
                if frame_sum is None:
                    frame_sum = np.zeros(frame_value.shape).astype('float64')
                frame_sum += frame_value
                count += 1
    except Exception as e:
        print(e)
        pass

    frame_mean = frame_sum / count
    assert(0. <= frame_mean.all() <= 1.)
    np.save(os.path.join(video_root_path, dataset, 'mean_frame_224.npy'), frame_mean)


def subtract_mean(dataset, video_root_path='/share/data/videos', is_combine=False):
    import os
    from skimage.io import imread
    import numpy as np
    from skimage.transform import resize

    frame_mean = np.load(os.path.join(video_root_path, dataset, 'mean_frame_224.npy'))
    training_combine = []
    testing_combine = []

    frame_path = os.path.join(video_root_path, dataset, 'training_frames')
    for frame_folder in os.listdir(frame_path):
        print('==> ' + os.path.join(frame_path, frame_folder))
        training_frames_vid = []
        for frame_file in sorted(os.listdir(os.path.join(frame_path, frame_folder))):
            frame_filename = os.path.join(frame_path, frame_folder, frame_file)
            frame_value = imread(frame_filename, as_grey=True)/256
            frame_value = resize(frame_value, (160, 240), mode='reflect')
            assert(0. <= frame_value.all() <= 1.)
            frame_value -= frame_mean
            training_frames_vid.append(frame_value)
        training_frames_vid = np.array(training_frames_vid)
        np.save(os.path.join(video_root_path, dataset, 'training_frames_{}.npy'.format(frame_folder[-3:])), training_frames_vid)
        if is_combine:
            training_combine.append(training_frames_vid)

    frame_path = os.path.join(video_root_path, dataset, 'testing_frames')
    for frame_folder in os.listdir(frame_path):
        print('==> ' + os.path.join(frame_path, frame_folder))
        testing_frames_vid = []
        for frame_file in sorted(os.listdir(os.path.join(frame_path, frame_folder))):
            frame_filename = os.path.join(frame_path, frame_folder, frame_file)
            frame_value = imread(frame_filename, as_grey=True)/256
            frame_value = resize(frame_value, (160, 240), mode='reflect')
            assert(0. <= frame_value.all() <= 1.)
            frame_value -= frame_mean
            testing_frames_vid.append(frame_value)
        testing_frames_vid = np.array(testing_frames_vid)
        np.save(os.path.join(video_root_path, dataset, 'testing_frames_{}.npy'.format(frame_folder[-3:])), testing_frames_vid)
        if is_combine:
            testing_combine.append(testing_frames_vid)
    if is_combine:
        training_combine = np.array(training_combine).reshape(-1, 160, 240, 1)
        testing_combine = np.array(testing_combine).reshape(-1, 160, 240, 1)
        np.save(os.path.join(video_root_path, dataset, 'training_frames_t0.npy'), training_combine)
        np.save(os.path.join(video_root_path, dataset, 'testing_frames_t0.npy'), testing_combine)


def build_h5(dataset, train_or_test, t, video_root_path='VIDEO_ROOT_PATH'):
    import h5py
    from tqdm import tqdm
    import os
    import numpy as np

    print("==> {} {}".format(dataset, train_or_test))

    def build_volume(train_or_test, num_videos, time_length):
        for i in tqdm(range(num_videos)):
            data_frames = np.load(os.path.join(video_root_path, '{}/{}_frames_{:03d}.npy'.format(dataset, train_or_test, i+1)))
            data_frames = np.expand_dims(data_frames, axis=-1)
            num_frames = data_frames.shape[0]

            data_only_frames = np.zeros((num_frames-time_length, time_length, 160, 240, 1)).astype('float16')

            vol = 0
            for j in range(num_frames-time_length):
                data_only_frames[vol] = data_frames[j:j+time_length] # Read a single volume
                vol += 1

            with h5py.File(os.path.join(video_root_path, '{0}/{1}_h5_t{2}/{0}_{3:02d}.h5'.format(dataset, train_or_test, time_length, i+1)), 'w') as f:
                if train_or_test == 'training':
                    np.random.shuffle(data_only_frames)
                f['data'] = data_only_frames

    os.makedirs(os.path.join(video_root_path, '{}/{}_h5_t{}'.format(dataset, train_or_test, t)), exist_ok=True)
    num_videos = len(os.listdir(os.path.join(video_root_path, '{}/{}_frames'.format(dataset, train_or_test))))
    build_volume(train_or_test, num_videos, time_length=t)


def combine_dataset(dataset, t, video_root_path='VIDEO_ROOT_PATH'):
    import h5py
    import os
    from tqdm import tqdm

    print("==> {}".format(dataset))
    output_file = h5py.File(os.path.join(video_root_path, '{0}/{0}_train_t{1}.h5'.format(dataset, t)), 'w')
    h5_folder = os.path.join(video_root_path, '{0}/training_h5_t{1}'.format(dataset, t))
    filelist = sorted([os.path.join(h5_folder, item) for item in os.listdir(h5_folder)])

    # keep track of the total number of rows
    total_rows = 0

    for n, f in enumerate(tqdm(filelist)):
      your_data_file = h5py.File(f, 'r')
      your_data = your_data_file['data']
      total_rows = total_rows + your_data.shape[0]

      if n == 0:
        # first file; create the dummy dataset with no max shape
        create_dataset = output_file.create_dataset('data', (total_rows, t, 160, 240, 1), maxshape=(None, t, 160, 240, 1))
        # fill the first section of the dataset
        create_dataset[:,:] = your_data
        where_to_start_appending = total_rows

      else:
        # resize the dataset to accomodate the new data
        create_dataset.resize(total_rows, axis=0)
        create_dataset[where_to_start_appending:total_rows, :] = your_data
        where_to_start_appending = total_rows

    output_file.close()


def preprocess_data(logger, dataset, t, video_root_path='VIDEO_ROOT_PATH'):
    import os

    # Step 1: Calculate the mean frame of all training frames
    # Check if mean frame file exists for the dataset
    # If the file exists, then we can skip re-generating the file
    # Else calculate and generate mean file
    logger.debug("Step 1/4: Check if mean frame exists for {}".format(dataset))
    mean_frame_file = os.path.join(video_root_path, dataset, 'mean_frame_224.npy')
    training_frame_path = os.path.join(video_root_path, dataset, 'training_frames')
    testing_frame_path = os.path.join(video_root_path, dataset, 'testing_frames')
    if not os.path.isfile(mean_frame_file):
        # The frames must have already been extracted from training and testing videos
        assert(os.path.isdir(training_frame_path))
        assert(os.path.isdir(testing_frame_path))
        logger.info("Step 1/4: Calculating mean frame for {}".format(dataset))
        calc_mean(dataset, video_root_path)

    # Step 2: Subtract mean frame from each training and testing frames
    # Check if training & testing frames are already been subtracted
    # If the file exists, then we can skip re-generating the file
    logger.debug("Step 2/4: Check if training/testing_frames_videoID.npy exists for {}".format(dataset))
    try:
        # try block will execute without AssetionError if all frames have been subtracted
        for frame_folder in os.listdir(training_frame_path):
            training_frame_npy = os.path.join(video_root_path, dataset, 'training_frames_{}.npy'.format(frame_folder[-3:]))
            assert(os.path.isfile(training_frame_npy))
        for frame_folder in os.listdir(testing_frame_path):
            testing_frame_npy = os.path.join(video_root_path, dataset, 'testing_frames_{}.npy'.format(frame_folder[-3:]))
            assert (os.path.isfile(testing_frame_npy))
    except AssertionError:
        # if all or some frames have not been subtracted, then generate those files
        logger.info("Step 2/4: Subtracting mean frame for {}".format(dataset))
        subtract_mean(dataset, video_root_path, t<=0)
    if t > 0:
        # Step 3: Generate small video volumes from the mean-subtracted frames and dump into h5 files (grouped by video ID)
        # Check if those h5 files have already been generated
        # If the file exists, then skip this step
        logger.debug("Step 3/4: Check if individual h5 files exists for {}".format(dataset))
        for train_or_test in ('training', 'testing'):
            try:
                h5_folder = os.path.join(video_root_path, '{}/{}_h5_t{}'.format(dataset, train_or_test, t))
                assert(os.path.isdir(h5_folder))
                num_videos = len(os.listdir(os.path.join(video_root_path, '{}/{}_frames'.format(dataset, train_or_test))))
                for i in range(num_videos):
                    h5_file = os.path.join(video_root_path, '{0}/{1}_h5_t{2}/{0}_{3:02d}.h5'.format(dataset, train_or_test, t, i+1))
                    assert(os.path.isfile(h5_file))
            except AssertionError:
                logger.info("Step 3/4: Generating volumes for {} {} set".format(dataset, train_or_test))
                build_h5(dataset, train_or_test, t, video_root_path)

        # Step 4: Combine small h5 files into one big h5 file
        # Check if this big h5 file is already been generated
        # If the file exists, then skip this step
        logger.debug("Step 4/4: Check if individual h5 files have already been combined for {}".format(dataset))
        training_h5 = os.path.join(video_root_path, '{0}/{0}_train_t{1}.h5'.format(dataset, t))
        if not os.path.isfile(training_h5):
            logger.info("Step 4/4: Combining h5 files for {}".format(dataset))
            combine_dataset(dataset, t, video_root_path)

    logger.info("Preprocessing is completed")

