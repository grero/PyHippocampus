import numpy as np
import matplotlib.pyplot as plt 

def plotFFT(data, fs):
	shape = data.shape
	def fftSingleTimeSeries(x, fs):
		nfft = 2.**(np.ceil(np.log(len(x)) / np.log(2)))
		fftx = np.fft.fft(x, int(nfft))
		numUniquePoints = int(np.ceil((nfft + 1) / 2))
		fftx = fftx[:numUniquePoints]
		mx = np.absolute(fftx)
		mx[1:len(mx) - 1] = mx[1:len(mx) - 1] * 2 
		mx = mx / (x.shape)[0]
		fn = fs / 2 
		f = [i*2*fn / nfft for i in range(numUniquePoints)]
		return mx, f
	if len(shape) > 1:
		fftProcessed = []
		for i in range(shape[1]):
			temp, f = fftSingleTimeSeries(data[i], fs)
			fftProcessed.append(temp)
		fftProcessed = np.array(fftProcessed)
		ax = plt.gca()
		mxMean = np.mean(fftProcessed)
		std = np.std(fftProcessed)
		ax.plot(f, mxMean + std)
		ax.plot(f, mxMean - std)
		return ax
	else:
		fftProcessed, f = fftSingleTimeSeries(data, fs)
		# ax = plt.gca()
		# ax.plot(f, fftProcessed)
		return fftProcessed, f

def removeLineNoise():
	pass 