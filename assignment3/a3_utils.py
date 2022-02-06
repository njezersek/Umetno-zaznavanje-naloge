import numpy as np
import cv2
from matplotlib import pyplot as plt

def draw_line(rho, theta, max_rho):
	a = np.cos(theta)
	b = np.sin(theta)

	x0 = a*rho
	y0 = b*rho

	x1 = int(x0 + max_rho*(-b))
	y1 = int(y0 + max_rho*(a))
	x2 = int(x0 - max_rho*(-b))
	y2 = int(y0 - max_rho*(a))

	plt.plot((y1,y2),(x1,x2),'r')