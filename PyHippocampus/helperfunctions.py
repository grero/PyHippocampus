import numpy as np
import numpy.matlib as nm

def plotFFT(data, fs):
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
	fftProcessed, f = fftSingleTimeSeries(data, fs)
	return fftProcessed[1:], f[1:]

def removeLineNoise(data, lineF, sampleF):
	pplc = int(np.fix(sampleF / lineF))
	isdims = isignal.shape
	if len(isdims) > 1:
		signal = isignal.flatten()
		slength = isdims[1]
	else:
		signal = isignal 
		slength = isdims[0]

	if slength < sampleF:
		cycles = int(np.fix(slength / pplc))
	else:
		cycles = lineF

	cpoints = cycles * pplc

	if cycles % 2 == 0:
		cplus = int(cycles / 2)
		cminus = int(cplus - 1) 
		pplus = int(cplus * pplc)
		pminus = int(cminus * pplc)
	else:
		cplus = int((cycles - 1 ) / 2)
		cminus = cplus 
		pplus = int(cplus * pplc) 
		pminus = pplus 

	indices = np.array(list(range(pplus+pplc, cpoints)) + list(range(0,slength)) + list(range(slength - cpoints, slength - (pminus+pplc))))

	mat_ind_ind = nm.repmat(np.arange(0, slength), cycles, 1) + pminus + nm.repmat(np.transpose(np.array([np.arange(-cminus, cplus+1)])) * pplc, 1, slength)

	mat_ind = indices[mat_ind_ind]
	mean_sig = np.mean(signal[mat_ind], axis = 0)
	osignal = signal - mean_sig 
	return osignal
