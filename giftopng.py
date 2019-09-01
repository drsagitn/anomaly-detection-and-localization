import os
from scipy.misc import imread, imsave
for file in os.listdir():
    a = imread(file)
    imsave(file+".png", a)

