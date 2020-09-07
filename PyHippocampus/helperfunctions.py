import numpy as np
import numpy.matlib as nm

def computeFFT(data, fs):
    # round up to power to 2
    nfft = 2.**(np.ceil(np.log(np.size(data,0)) / np.log(2)))
    # compute FFT in columns
    fftx = np.fft.rfft(data, int(nfft), 0)
    # convert complex numbers to magnitude
    mx = np.absolute(fftx)
    # return the frequencies to make it easier to plot the results
    fn = fs / 2
    nfft2 = nfft / 2
    f = np.arange(0,nfft2+1,1) * fn / nfft2
    return mx, f 

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
