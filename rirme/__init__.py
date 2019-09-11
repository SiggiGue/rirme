import logging as _logging
import pathlib as _pathlib
import shutil as _shutil
import uuid as _uuid
import yaml as _yaml

from . import conf as _cfg


_logger = _logging.getLogger(_cfg.LOGGERNAME)


def init(name=None, path='.'):
    """Initializes a rirmeasuring project in given `path`
    
    Parameters
    ----------
    name : str
        Project name.
    path : str or pathlib.Path
        Path where to create the config.
    """
    path = _pathlib.Path(path)
    name = name or path.parent.name
    cfpath = (path/_cfg.CONFIGFILENAME).absolute()
    if cfpath.exists():
        raise FileExistsError(f'{cfpath} already exists. Aborted.')
    _shutil.copy(_cfg.USERCONFIGFILEPATH, path)
    _logger.info('Initializing rirmeasuring project ...')
    _logger.info(f'Created {_cfg.CONFIGFILENAME}')
    with open(path/_cfg.PROJECTFILENAME, 'w') as fp:
        _yaml.safe_dump({
            'name': name,
            'description': '',
            'source': '',
            'receiver': '',
            'room': '',
            'uid': _uuid.uuid4().hex
        }, fp)
    _logger.info(f'Created {_cfg.PROJECTFILENAME}')
    _logger.info('Done.')


def new(name, path='.'):
    """Create a new measurement in a folder called `name` in `path`
    
    Parameters
    ----------
    name : str
        Project name.
    path : str or pathlib.Path
        Path where to create the project folder.

    """
    path = (_pathlib.Path(path)/name)
    _logger.info(f'Creating new project folder {path}')
    path.mkdir(parents=True)
    init(name, path=path)