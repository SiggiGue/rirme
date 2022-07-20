import logging as _logging
import pathlib as _pathlib
import shutil as _shutil
import warnings as _warnings

import yaml as _yaml

from .__version__ import __version__


PROJECTFILENAME = 'project.rirme.yml'
CONFIGFILENAME = 'config.rirme.yml'
CONFIGDIR = '.rirmeasuring'
MODULEPATH = _pathlib.Path(__file__).parent.absolute()
USERPATH = _pathlib.Path('~').expanduser()
USERCONFIGDIR = USERPATH/CONFIGDIR
USERCONFIGFILEPATH = USERCONFIGDIR/CONFIGFILENAME
H5FILEEXT = 'rirme.h5'
LOGGERNAME = 'rirme'


_logging.basicConfig(format='%(levelname)s: %(message)s', level=_logging.INFO)
logger = _logging.getLogger(LOGGERNAME)


def _check_config_version(config):
    config_version = config['rirmeasuring']['version']
    if config_version != __version__:
        _warnings.warn(
            f'Your config Version {config_version}'
            f' is not matching with rirmeasuring.__version__ {__version__}. '
            f'To get the defaults for version {__version__}'
            f' rename {USERCONFIGFILEPATH} and reimport this module.')
    return config


def load_config(filepath):
    """Returns a config dictionary from given `filepath`."""
    filepath = _pathlib.Path(filepath)
    if filepath.is_dir():
        filepath = filepath / CONFIGFILENAME
    with open(filepath, 'r') as fp:
        return _check_config_version(dict(_yaml.safe_load(fp)))
    
    
def init_userconfig():
    """Checks for USERCONFIGDIR and creates default config files."""
    if not USERCONFIGDIR.exists():
        USERCONFIGDIR.mkdir(exist_ok=True, parents=True)
    if not USERCONFIGFILEPATH.exists():
        _shutil.copy(MODULEPATH/CONFIGFILENAME, USERCONFIGDIR)
        

def load_defaults():
    """Returns default for the USER.
    A folder _USERCONFIGDIR will be created if not present and filled with default configs.
    """
    init_userconfig()
    return load_config(USERCONFIGFILEPATH)
        