from setuptools import setup

setup(
    name='rirme',
    version='0.1.1',
    author='Siegfried Gündert',
    author_email='siegfried.guendert-a-gmail.com',
    license='MPL-2.0',
    packages=['rirme'],
    entry_points = {
        'console_scripts': ['rirme=rirme.shell:cli'],
    }
)
