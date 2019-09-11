import numpy as _np
import matplotlib.pyplot as _plt
import scipy.signal as _signal

class Analyzer(object):
    # TODO: this should go into a distinct riranalyzer package
    def __init__(self, res, name=None):
        self._res = res
        self._name = name or res.uid
        self._samplerate = self._res.samplerate
        self._dt = 1/self._samplerate
        self._htrirs = _signal.hilbert(self._res.rirs)

    def plot_all(self, savefmt=None, dpi=250):
        def savefig():
            if savefmt:
                _plt.savefig('.'.join((self._name, 'timesignal', savefmt)), format=savefmt, dpi=dpi)

        _plt.figure()
        self.plot_timesignal()
        savefig()
        _plt.figure()
        self.plot_reflectogram()
        savefig()

    def _annotate(self, ax=None):
        ax = ax or _plt.gca()
        _plt.annotate(self._name, (0, 0), 
            xytext=(ax.get_xlim()[0], ax.get_ylim()[0]),
            fontsize='small')
        
    def _legend_channels(self):
        _plt.legend([f'CH{i}' for i in range(self._res.rirs.shape[1])])

    def plot_timesignal(self):
        time = _np.arange(0, self._res.rirs.shape[0]*self._dt, self._dt)
        _plt.plot(time, self._res.rirs)
        _plt.title('RIR Time Signal')
        self._annotate()
        _plt.ylabel('Amplitude (FS)')
        _plt.xlabel('Time /s')
        _plt.tight_layout()
        self._legend_channels()

    def plot_reflectogram(self):
        time = _np.arange(0, self._res.rirs.shape[0]*self._dt, self._dt)
        _plt.plot(time, _np.abs(self._htrirs)**2)
        _plt.title('RIR Reflectogram')
        self._annotate()
        _plt.ylabel('Amplitude (FS)')
        _plt.xlabel('Time /s')
        _plt.tight_layout()
        self._legend_channels()