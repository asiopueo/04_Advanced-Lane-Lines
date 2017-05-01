import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

from moviepy.editor import VideoFileClip



def process_image(img):
	output_image = np.copy(img)
	return output_image



clip = VideoFileClip('./videos/project_video.mp4')

output_file = 'test_video.mp4'

output_stream = clip.fl_image(process_image)
output_stream.write_videofile(output_file, audio=False)







