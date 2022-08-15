import dataclasses as _dataclasses
import logging as _logging
import pathlib as _pathlib
import glob as _glob
import queue as _queue
import shutil as _shutil
import time as _time
import uuid as _uuid
import warnings as _warnings

import yaml as _yaml
import numpy as _np
import scipy.io.matlab as _mat
import scipy.signal as _signal
import matplotlib.pyplot as _plt
import hdfdict as _hdfdict
import sounddevice as _sd
import soundfile as _sf

from gccestimating import GCC
from syncsweptsine import hannramp
from syncsweptsine import SyncSweep
from syncsweptsine import HigherHarmonicImpulseResponse
from syncsweptsine import LinearModel

from . import __version__
from . import conf as _cfg
from .containers import Result


_logger = _logging.getLogger(_cfg.LOGGERNAME)

devices = _sd.query_devices


DEFAULTS = _cfg.load_defaults()


def create_normalized_white_gaussian_noise(samples, channels=None):
    """Returns normalized white gaussian noise

    Parameters
    ----------
    samples : int
        Number of samples.
    channels : int
        Number of channels. default is None.

    Returns
    -------
    noise : ndarray
        Numpy array with white gaussian noise of shape (`samples`, `channels`)
        if channels is None, shape will be (`samples`,)
    
    """
    if channels:
        noise = _np.random.randn(samples, channels)
    else:
        noise = _np.random.randn(samples)
    return noise/_np.max(_np.abs(noise))


def take_channels(data, channelmap):
    """Returns a slice of data for given `channelmap`
    
    Parameters
    ----------
    data : ndarray
    channelmap : iterable
        Channel mapping starting from 1.
    manifold to
    Returns
    -------
    data : ndarry
        Sliced data of shape (len(`data`), len(`channelmap`))

    Notes
    -----
    First dimension of `data` is considered to be samples axis
    and the second dimension is the channel axis.

    """
    channelmap = [c-1 for c in channelmap]
    return data[:, channelmap]


def manifold_to_channels(data, channels, channelmap):
    """Returns manifolded `data` into an ndarray of channels for given channelmap.

    Parameters
    ----------
    data : ndarray
        Data array, one dimensional.
    channels : int
        Number of channels.
    channelmap : iterable
        Map of channels where data should be copied to.

    Returns
    -------
    out : ndarray
        Array of shape (len(data), channels)
        where channemap channels are filled with the data
        and all remaining channels are zero.

    """
    channelmap = [c-1 for c in channelmap]
    out = _np.zeros((len(data), channels), dtype=data.dtype)
    if len(data.shape) == 1:
        data = _np.tile(_np.atleast_2d(data), (len(channelmap), 1)).T
    out[:, channelmap] = data
    return out


def genblocks(sig, blocksize):
    """Yields signal blocks from `sig` of size `blocksize`.

    Parameters
    ----------
    sig : ndarray
    blocksize : int

    Yields
    ------
    block : ndarray
        array of shape (blocksize, ...)

    """
    siglen = len(sig)
    for index in range(0, siglen, blocksize):
        if index > siglen-blocksize:
            shape = list(sig.shape)
            shape[0] = blocksize
            x = _np.zeros(shape, dtype=sig.dtype)
            x[:siglen-index, ...] = sig[index:, ...]
            yield x
        else:
            yield sig[index:index+blocksize, ...]


class SounddeviceQueueCallback(object):
    """Returns a Queue Callbac callable object 
    usable by sounddevice Streams.

    Parameters
    ----------

    """

    def __init__(self, input_queue=None, output_queue=None):
        self.input_queue = input_queue or _queue.Queue() 
        self.output_queue = output_queue or _queue.Queue()
    
    def __call__(self, indata, outdata, frames, time, status):
        if status:
            print(status)
        self.input_queue.put(indata.copy())
        try:
            outdata[:] = self.output_queue.get_nowait()
        except _queue.Empty:
            outdata.fill(0)


def estimate_delay(gcc_estimator, samplerate=None):
    """Returns a delay estimate for given gcc_estimator.
    
    Parameters
    ----------
    gcc_estimator : GCC.Estimate instance
    
    Returns
    -------
    delay : int

    """
    return -gcc_estimator.index_to_lag(
        index=_np.argmax(_np.abs(gcc_estimator.sig)), 
        samplerate=samplerate)


def estimate_adaptive_delay_and_decay(
        input_queue, 
        output_queue, 
        samplerate, 
        channels,
        channelmap_input, 
        channelmap_output,
        blocksize, 
        noise_level_dbfs,
        noise_duration,
        maxdecaytime_sec=None,
        bgnoiseblocks=10,
        dtype=None,
        **kwargs):

    """Returns raw data, delay and decay estimate.

    Parameters
    ----------

    Returns
    -------


    """
    _logger.info('Estimating delays and decay...')
    noise_normalized = create_normalized_white_gaussian_noise(
        int(noise_duration*samplerate)).astype(dtype)
    noise = 10**(noise_level_dbfs/20) * hannramp(noise_normalized, blocksize, blocksize)
    noise_multichannel = manifold_to_channels(noise, channels, channelmap_output)

    bgnoise = _np.concatenate([input_queue.get() for i in range(bgnoiseblocks)])
    bgnoise = take_channels(bgnoise, channelmap_input)
    bgnoisesq = bgnoise*bgnoise
    maxmedbgnoisesq = _np.max(_np.median(bgnoisesq, 0))

    bgnoiselevels = 10*_np.log10(_np.mean(bgnoisesq, 0))
    bgnoiselevel = '|'.join(('{:6.1f}'.format(lbg) for lbg in bgnoiselevels))
    _logger.info(f'Background noise level ({bgnoiselevel}) dB (FS)')
    
    maxiter = (maxdecaytime_sec*samplerate)//blocksize
    count = 0
    datalist = []
    finished = False
    noise_was_there = False
    levels = []
    for block in genblocks(noise_multichannel, blocksize):
        output_queue.put(block)
    
    while not finished:
        data = input_queue.get()
        data = take_channels(data, channelmap_input)
        datalist += [data]
        datasq = data*data

        if _logger.level < _logging.WARNING:
            levels = max(levels, list(10*_np.log10(_np.mean(datasq, 0))))
            level = '|'.join(('{:6.1f}'.format(lv) for lv in levels))
            print(f'INFO: Max level: ({level}) dB (FS)', end='\r')
        
        maxmeddatasq = _np.max(_np.median(datasq, 0))
        
        if maxmeddatasq >= maxmedbgnoisesq*10:  # at least 10 dB SNR
            noise_was_there = True

        if noise_was_there and (maxmeddatasq < maxmedbgnoisesq):
            finished = True

        if count >= maxiter:
            print('Max iterations for delay estimation reached. '
                'Check maxiter setting or check the SNR in your measurement setup.')
            break

        count += 1
    if _logger.level < _logging.WARNING:
        print('')

    snr = '|'.join((('{:.1f}'.format(r) for r in (_np.array(levels) - _np.array(bgnoiselevels)))))
    _logger.info(f'SNR ({snr}) dB (FS)')

    data = _np.concatenate(datalist) 
    gcc_gen= (GCC(noise_normalized, data[:, c]) for c in range(len(channelmap_input)))
    delays = [estimate_delay(gcc.phat()) for gcc in gcc_gen]
    delays_sec = [_np.round(d/samplerate, 4) for d in delays]
    decay = int(1.75*(blocksize*count  # all samples
        - (len(noise) - 2*blocksize))) # minus noise signal samples on its own without hannig flanks
    decay_sec = decay / samplerate
    _logger.info(f'Channel Delays: {delays_sec} s')
    _logger.info(f'Decay estimate: {decay_sec:.3f} s')
    return noise, data, delays, decay


def estimate_lm_from_sweeps(refsweep, measured_sweeps, irlength, delays, window=None):
    """Yields linear models for given sweeps."""
    for channel in range(measured_sweeps.shape[1]):
        hhir = HigherHarmonicImpulseResponse.from_sweeps(
            syncsweep=refsweep, 
            measuredsweep=measured_sweeps[:, channel])
        irlength = min(hhir.max_hir_length(1), irlength)
        linear_model = LinearModel.from_higher_harmonic_impulse_response(
            hhir=hhir, 
            length=irlength,
            delay=delays[channel],
            window=window)
        yield linear_model


class Sweep(SyncSweep):
    @classmethod
    def from_config(cls, config):
        return cls(
            startfreq=config['sweep']['startfreq'],
            stopfreq=config['sweep']['stopfreq'],
            durationappr=config['sweep']['durationappr'],
            samplerate=config['measurement']['samplerate'],
        )


class Measurement(object):
    def __init__(self, config):
        self._config = config or DEFAULTS
        self._sweep = None

    def save_config(self, path):
        with open(_pathlib.Path(path)/_cfg.CONFIGFILENAME, 'w') as fp:
            return _yaml.dump(fp, self._config)

    @classmethod
    def from_config_file(cls, filepath):
        config = _cfg.load_config(filepath)
        return cls(**config)

    def run_adaptive_delay_and_decay_measurement(self):
        cfg = {
            **self._config, 
            **self._config['measurement'], 
            **self._config['sounddevice']
            }

        stream_kwargs = {k: cfg[k] for k in (
            'samplerate', 'device', 'channels', 'dtype', 'blocksize')}
        callback = SounddeviceQueueCallback()
        with _sd.Stream(
                callback=callback,
                **stream_kwargs):
            noise, data, delay, decay = estimate_adaptive_delay_and_decay(
                input_queue=callback.input_queue, 
                output_queue=callback.output_queue,
                noise_level_dbfs=cfg['noise']['level_dbfs'],
                noise_duration=cfg['noise']['duration'], 
                **cfg)
        return noise, data, delay, decay

    def run_sweep_measurement(self):
        cfg = {
            **self._config, 
            **self._config['sweep'], 
            **self._config['measurement'],
            **self._config['sounddevice']}
        if not self._sweep:
            self._sweep = SyncSweep(
                startfreq=cfg['startfreq'],
                stopfreq=cfg['stopfreq'],
                durationappr=cfg['durationappr'],
                samplerate=cfg['samplerate'],
            )

        sweep_playback_signal = self._sweep.get_windowed_signal(
            left=cfg['flanksamples'],
            right=cfg['flanksamples'],
            pausestart=cfg['pausestart'],
            pausestop=cfg['pausestop'],
            amplitude=10**(cfg['sweep']['level_dbfs']/20))

        measured_sweep = _sd.playrec(
            sweep_playback_signal, 
            samplerate=cfg['samplerate'], 
            device=cfg['device'],
            input_mapping=cfg['channelmap_input'], 
            output_mapping=cfg['channelmap_output'], 
            blocking=True)
        return sweep_playback_signal, measured_sweep

    def run(self):
        cfg = {
            **self._config, 
            **self._config['measurement']}
        auto = cfg['auto']
        delays = cfg['delays']
        irlength = cfg['irlength']
        pausestart = cfg['pausestart']
        pausestop = cfg['pausestop']
        window = cfg['window']
        _time.sleep(cfg['wait_before_start_sec'])
        if auto:
            (noise, 
            measured_noises, 
            delays,
            decay) = self.run_adaptive_delay_and_decay_measurement()
            pausestop = max(pausestop, decay)
            self._config['measurement']['pausestop'] = pausestop
            irlength = max(irlength, decay)
        else:
            measured_noises = None
        sweep, measured_sweeps = self.run_sweep_measurement()
        delays_total = _np.array(delays) + pausestart
        delays_total[:] = _np.median(delays_total).astype(_np.int)
        linear_models = estimate_lm_from_sweeps(self._sweep, measured_sweeps, irlength, delays_total, window)
        rirs = _np.array([lm.kernel.ir for lm in linear_models]).transpose()

        res = Result(
            uid=_uuid.uuid4().hex,
            config=self._config,
            input_sweep=sweep,
            input_noise=noise,
            measured_sweeps=measured_sweeps, 
            measured_noises=measured_noises,
            delays=delays,
            delays_total=delays_total,
            samplerate=self._config['measurement']['samplerate'],
            rirs=rirs)
        return res


def run(path='.', repetitions=1, prefix=''):
    """Runs a RIR MEasurement

    Parameters
    ----------
    path : str or pathlib.Path
        path to a directory or path to a configfile
        if a directory path is provided, it is supposed to contain a config file.
        
    repetitions : int
        Number of measurement repetitions.
        Default is 1.
    
    prefix : str
        Prefix the name of the result file

    Returns
    -------
    result : Result
    
    """
    path = _pathlib.Path(path)
    if path.is_dir():
        path = path/_cfg.CONFIGFILENAME
    if not path.exists():
        raise FileNotFoundError(f'{path} could not be found. See methods `new` and `init` to create default configs.')
    _logger.info(f'Run measurement using {path}')
    config = _cfg.load_config(path)
    _logger.debug(f'Config loaded: {config}')
    mmt = Measurement(config)
    _logger.debug(f'Created measurement instance {mmt}')

    def run_and_save(num):
        _logger.info(f'Start of measurement {num}. \nPlease be quiet!')
        res = mmt.run()
        if prefix:
            fname = f'{prefix}_m{num}.{res.uid}.{_cfg.H5FILEEXT}'
        else:
            fname = f'm{num}.{res.uid}.{_cfg.H5FILEEXT}'
        fpath = res.save(path=path.parent, fname=fname)
        _logger.info(f'Saved measruement: {fpath}')
        return res

    results = [run_and_save(r) for r in range(repetitions)]
    _logger.info('Measurement finished.')
    return results
