import numpy as np
import cv2, glob
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def drawEllipse(mu, cov, n_std=1):
	# input:
	# mu: the mean value of the data
	# cov: the covariance matrix of the data
	# n_std: sigma value for the size of the ellipse

	# source: https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html

	pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
	ell_radius_x = np.sqrt(1 + pearson)
	ell_radius_y = np.sqrt(1 - pearson)
	ellipse = Ellipse((0,0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='none', edgecolor='lime')

	scale_x = np.sqrt(cov[0, 0]) * n_std
	scale_y = np.sqrt(cov[1, 1]) * n_std

	transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mu[0],mu[1])

	ellipse.set_transform(transf + plt.gca().transData)

	plt.gca().add_patch(ellipse)