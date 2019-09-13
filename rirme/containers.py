import dataclasses as _dataclasses
import pathlib as _pathlib

import numpy as _np

import hdfdict as _hdfdict
import soundfile as _sf
import scipy.io.matlab as _mat

from . import conf as _cfg


@_dataclasses.dataclass
class Result:
    uid: str
    config: dict
    input_sweep: _np.ndarray
    input_noise: _np.ndarray
    measured_sweeps: _np.ndarray
    measured_noises: _np.ndarray
    delays: _np.ndarray
    delays_total: _np.ndarray
    samplerate: float
    rirs: _np.ndarray

    def save(self, path='.', fname=''):
        if not fname:
            fname = '.'.join((self.uid, _cfg.H5FILEEXT))
        fpath = str(_pathlib.Path(path)/fname)
        _hdfdict.dump(
            _dataclasses.asdict(self), 
            fpath
            )
        return fpath

    @classmethod
    def load(cls, filepath):
        return cls(**_hdfdict.load(filepath, lazy=False))

    def export_soundfiles(self, format, path='.'):
        kwargs = dict(
            samplerate=self.samplerate,
            format=format
            )
        path = _pathlib.Path(path)
        filepath = path/'.'.join((self.uid, 'measured_sweeps', format))
        _sf.write(file=filepath, data=self.measured_sweeps, **kwargs)
        if _np.any(self.measured_noises):
            filepath = path/'.'.join((self.uid, 'measured_noises', format))
            _sf.write(file=filepath, data=self.measured_noises, **kwargs)

    def export_matfile(self, path='.', fname=''):
        if not fname:
            fname = '.'.join((self.uid, 'rirme', 'mat'))
        fpath = str(_pathlib.Path(path)/fname)
        _mat.savemat(fpath, _dataclasses.asdict(self))
        return fpath