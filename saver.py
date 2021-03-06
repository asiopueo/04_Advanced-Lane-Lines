import os
import matplotlib.pyplot as plt


def image_saver(img, filename='llines', directory='./output_images'):
	itemlist = os.listdir(directory)
	cntr = 0
	
	for item in itemlist:
		prefix = item.split('.')[0]
		extension = item.split('.')[1]
		if prefix.split('_')[0] == filename and extension == 'png':
			fnd_cntr = int(prefix.split('_')[1])
			if fnd_cntr >= cntr:
				cntr = fnd_cntr + 1
		
	cntr_str = str(cntr).zfill(2)
	output_handle = directory + '/' + filename + '_' + cntr_str + '.png'
	
	plt.imsave(output_handle, img)

	return
	



def video_saver(stream, filename='llines', directory='./output_videos'):
	itemlist = os.listdir(directory)
	cntr = 0
	
	for item in itemlist:
		prefix = item.split('.')[0]
		extension = item.split('.')[1]
		if prefix.split('_')[0] == filename and extension == 'mp4':
			fnd_cntr = int(prefix.split('_')[1])
			if fnd_cntr >= cntr:
				cntr = fnd_cntr + 1
		
	cntr_str = str(cntr).zfill(2)
	output_handle = directory + '/' + filename + '_' + cntr_str + '.mp4'
	
	stream.write_videofile(output_handle, audio=False)

	return





