import click
from . import conf as _cfg


_logger = _cfg.logger

@click.group()
@click.option('--loglevel', default='INFO', help='Set the logging level {[INFO], DEBUG, WARNING, ERROR, CRITICAL}.')
def cli(loglevel):
    """
    Room Impulse Response MEasuring
    """
    _logger.setLevel(getattr(_cfg._logging, str(loglevel).upper()))


@cli.command('init')
@click.option('--name', default=None)
@click.option(
    '--path', '-p', default='.', 
    help='Path where the config and project files are placed.')
def _init(name, path):
    """Initialize rirmeasuring project in path."""
    from . import init
    return init(name=name, path=path)


@cli.command('new')
@click.argument('name')
@click.option(
    '--path', '-p', default='.', 
    help='Path where the rirmeasuring project folder will be created.')
def new(name, path):
    """Create new rirmeasuring project folder and initialize it."""
    from . import new
    return new(name=name, path=path)


@cli.command('run')
@click.option('--path', '-p', default='.')
@click.option('--prefix', '-x', default='',
    help='Prefix for result file(s).')
@click.option('--repetitions', '-r', default=1)
@click.option('--mat', '-m', is_flag=True, default=False)
@click.option('--snd', '-s', 
    default=None, 
    help='Export soundfiles using soundfile library. To see supported formats run:' 
        ' `python -c "import soundfile;print(soundfile.available_formats().keys())"`')
def run(path, prefix, repetitions, mat, snd):
    """Run RIR MEasurement"""
    from . import measuring
    results = measuring.run(
        path=path, 
        repetitions=repetitions, 
        prefix=prefix)
    if mat or snd:
        for res in results:
            if mat:
                res.export_matfile(path=path)
                _logger.info('Exported matfile.')
            if snd:
                res.export_soundfiles(format=snd, path=path)
                _logger.info(f'Exported {snd}.')


@cli.command('devices')
@click.option('--device', '-d', default=None)
@click.option('--kind', '-k', default=None)
def _list_sounddevices(device, kind):
    """List available sound devices."""
    from sounddevice import query_devices
    print(query_devices(device=device, kind=kind))


@cli.command('analyze')
@click.argument('name', type=click.Path(), default='.'.join(('*', _cfg.H5FILEEXT)))
@click.option('--save', '-s', default='png')
@click.option('--dpi', '-d', default=250)
@click.option('--path', '-p', default='.')
@click.option('--show', default=True, is_flag=True)
def _analyze(name, path, save, dpi, show):
    """Analyze measurement results. 
    NAME is either a explicit filename or a part of the filename(s) that will be globbed."""
    import pathlib as _pathlib
    import glob as _glob
    from .analyzing import Analyzer
    from .containers import Result
    from matplotlib import pyplot as _plt

    path = _pathlib.Path(path)
    filepath = path/name

    if not _glob.has_magic(name) and not filepath.exists():
        filepaths = path.glob(f'*{name}*.{_cfg.H5FILEEXT}')
    elif _glob.has_magic(str(filepath)):
        filepaths = path.glob(name)
    else:
        filepaths = [filepath]

    for filepath in filepaths:
        _logger.info(f'Creating analysis plots for {filepath}')
        analyzer = Analyzer(Result.load(str(filepath)), name=filepath.name)
        analyzer.plot_all(save, dpi)
    if show:
        _plt.show()


@cli.command('convolve')
@click.argument('name', type=click.Path(), default='.'.join(('*', _cfg.H5FILEEXT)))
@click.option('--input', '-i', default='')
@click.option('--save', '-s', default='')
@click.option('--path', '-p', default='.')
@click.option('--quiet', '-q', default=False, is_flag=True)
@click.option('--level', '-l', default=-10)
def _convolve(name, input, path, save, quiet, level):
    """Convolve and listen to measurement results. """
    import pathlib as _pathlib
    import glob as _glob
    import sounddevice as _sd
    import soundfile as _sf
    import numpy as np
    from scipy.signal import convolve
    from .containers import Result
    from matplotlib import pyplot as _plt

    if not save and quiet:
        raise ValueError('At least listen to it or save it (e.g. -s flac)')
    path = _pathlib.Path(path)
    filepath = (path/name).absolute()
    
    if not _glob.has_magic(name) and not filepath.exists():
        filepaths = path.glob(f'*{name}*.{_cfg.H5FILEEXT}')
    elif _glob.has_magic(str(filepath)):
        filepaths = path.glob(filepath)
    else:
        filepaths = [filepath]
    
    for filepath in filepaths:
        _logger.info(f'Processing {filepath}')

        res = Result.load(str(filepath))
        rirs = res.rirs
        rirs = rirs/np.max(np.abs(rirs))
        if not input:
            print(rirs.shape, rirs.max())
            out = rirs
        elif input == 'noise':
            out = convolve(rirs, np.random.randn(res.samplerate, rirs.shape[1]))
        else:
            data, samplerate = _sf.read(input)
            if samplerate != res.samplerate:
                raise ValueError(f'Samplerates do not match (IR:{res.samplerate})(soundfile:{samplerate})')
            else:
                out = convolve(rirs, data)
                if not quiet:
                    _logger.info('Listening to: Input signal...')
                    _sd.play(10**(level/20)*data/np.max(np.abs(data)), samplerate=res.samplerate, blocking=True)

        if not quiet: 
            out = 10**(level/20) * out/np.max(np.abs(out))
            _logger.info('Listening to: Output Signal')
            _sd.play(out, samplerate=res.samplerate, blocking=True)

        if save:
            print(save)