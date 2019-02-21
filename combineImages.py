from PIL import Image


def combine2Images(srcPath, desPath, index):
    images = [Image.open(srcPath.format(index)), Image.open(desPath.format(index))]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        y_offset = int((max_height - im.size[1])/2)
        new_im.paste(im, (x_offset,y_offset))
        x_offset += im.size[0]
    os.makedirs('combineImages', exist_ok=True)
    new_im.save('combineImages/combine{0:03d}.jpg'.format(index))

def changeImageFormat(formatType, inputPath, index):
    from skimage.io import imread, imsave
    images = imread(inputPath.format(index))
    os.makedirs('newformatedImages', exist_ok=True)
    imsave('newformatedImages/formated{0:03d}.{1}'.format(index, formatType))


def changeImageFormat(formatType, inputPath, index):
    import numpy

    images = Image.open(inputPath.format(index))
    new_im = Image.fromarray(numpy.array(images).astype(numpy.uint8))
    os.makedirs('newformatedImages', exist_ok=True)
    new_im.save('newformatedImages/formated{0:03d}.{1}'.format(index, formatType))

import os
for i in range(200):
    src = os.path.join('VIDEO_ROOT_PATH/UCSD_ped1/testing_frames/Test019', '{0:03d}.tif')
    # des = os.path.join('logs/UCSD_ped1/jobs/3df36c94-5457-4bc3-a8b9-7af636acb134/result', 'UCSD_ped1_err_vid19_frm{0:03d}.png')
    # combine2Images(src, des, i+1)

    changeImageFormat("jpg", src, i+1)

