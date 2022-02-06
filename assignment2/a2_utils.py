import numpy as np
import cv2
from matplotlib import pyplot as plt

def read_data(filename):
	# reads a numpy array from a text file
	with open(filename) as f:
		s = f.read()

	return np.fromstring(s, sep=' ')

def gauss_noise(I, magnitude):
	# input: image, magnitude of noise

	return I+np.random.rand(I.shape[0], I.shape[1])*magnitude

def sp_noise(I, percent):
	# input: image, percent of corrupted pixels
	res = I.copy()

	res[np.random.rand(I.shape[0], I.shape[1])<percent]=255
	res[np.random.rand(I.shape[0], I.shape[1])<percent]=0

	return res