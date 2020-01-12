import time
from runMe import *
from busProjectTest import runTest
import sys
import os


myAnnFileName = 'annotationsTrain_test_final_3.txt'
busDir = 'buses'
saveDir = 'result_final_1'
runTest('annotationsTrain.txt', myAnnFileName, busDir, saveDir, elapsed=10)


import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

