import numpy as np 
from scipy import signal 
import matplotlib.pyplot as plt 

def resampleData(analogData, samplingRate, resampleRate):
	numberOfPoints = int(resampleRate * (len(analogData) / samplingRate))
	analogData = signal.resample(analogData, numberOfPoints)
	return analogData

def lowPassFilter(analogData, samplingRate = 30000, resampleRate = 1000, lowFreq = 1, highFreq = 150, LFPOrder = 8, padlen = 0, display = False, saveFig = False):
	analogData = analogData.flatten()
	lfpsData = resampleData(analogData, samplingRate, resampleRate)
	fn = resampleRate / 2
	lowFreq = lowFreq / fn 
	highFreq = highFreq / fn 
	sos = signal.butter(LFPOrder, [lowFreq, highFreq], 'bandpass', fs = resampleRate, output = "sos")
	print("Applying low-pass filter with frequencies {} and {} Hz".format(lowFreq * fn, highFreq * fn))
	lfps = signal.sosfiltfilt(sos, lfpsData, padlen = padlen)
	if display: 
		lfpPlot(analogData, lfpsData, lfps, saveFig = saveFig)
		print('saved figure')
	return lfps, resampleRate

def highPassFilter(analogData, samplingRate = 30000, lowFreq = 500, highFreq = 7500, HPOrder = 8, padlen = 0, display = False, savefig = False):
	fn = samplingRate / 2
	lowFreq = lowFreq / fn 
	highFreq = highFreq / fn 
	sos = signal.butter(HPOrder, [lowFreq, highFreq], 'bandpass', fs = samplingRate, output = "sos")
	print("Applying high-pass filter with frequencies {} and {} Hz".format(lowFreq * fn, highFreq * fn))
	hps = signal.sosfiltfilt(sos, analogData, padlen = padlen)
	if display: 
		highpassPlot(analogData, hps, lowFreq = lowFreq * fn, highFreq = highFreq * fn saveFig = False)
	return hps, samplingRate

def lfpPlot(originalData, resampledData, filteredData, saveFig, samplingRate = 30000, resampleRate = 1000):
	fig, (ax1, ax2) = plt.subplots(2, 1)
	ax1.plot(originalData)
	ax1.set_title("Original Data at Sampling Rate of {} Hz".format(samplingRate))
	ax2.plot(resampledData, color = 'blue', label = 'Resampled')
	ax2.plot(filteredData, color = 'red', label = 'Filtered')
	ax2.set_title("Resampled Data at Sampling Rate of {} Hz".format(resampleRate))
	ax2.legend()
	plt.tight_layout()
	if saveFig: 
		fig.savefig('lfp-plot.png')
	return 

def highpassPlot(originalData, filteredData, lowFreq = 500, highFreq = 7500,saveFig):
	fig = plt.figure()
	plt.plot(originalData, color = 'blue', label = 'Original')
	plt.plot(filteredData, color = 'red', label = 'Filtered')
	plt.set_title('High pass filtered data between {} and {} Hz'.format(lowFreq, highFreq))
	plt.legend()
	plt.tight_layout()
	if savefig:
		fig.savefig('highpass-plot.png')
	return 