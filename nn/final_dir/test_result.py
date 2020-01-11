import time
from runMe import *
from busProjectTest import runTest
import sys
import os

myAnnFileName = 'annotationsTrain_test_3.txt'
busDir = 'buses'
saveDir = 'result_2'
runTest('annotationsTrain.txt', myAnnFileName, busDir, saveDir, elapsed=10)
