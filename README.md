# Room Impulse Response Measuring and Evaluation

This package implements 

-  a room impulse response measurement procedure using
   Synchronized Swept Sines, Generalized Cross Correlation and some automation functionality.

- Analysing and evaluation tools (e.g. convolving and listening to measurements and soundsamples.)

## Workflow
This tool can be used as follows

```shell 
$ rirme new example-measurement
INFO: Creating new project folder example-measurement
INFO: Initializing rirmeasuring project ...
INFO: Created config.rirme.yml
INFO: Created project.rirme.yml
INFO: Done.

$ cd example-measurement
$ ls 
config.rirme.yml*  project.rirme.yml*
```
Now edit the yml files. 
Run a measurement:

```shell
$ rirme run

INFO: Run measurement using config.rirme.yml
INFO: Start of measurement 0. 
Please be quiet!
INFO: Estimating delays and decay...
INFO: Background noise level ( -60.9) dB (FS)
INFO: SNR (35.4) dB (FS)
INFO: Channel Delays: [0.1993] s
INFO: Decay estimate: 1.217 s
INFO: Saved measruement: 4cbd86acd28741d4a51812745250af62.rirme.h5
INFO: Measurement finished.

```
For further commands see the command line interface help outputs  below.


## Command Line Interface

```shell
$ rirme --help
Usage: rirme [OPTIONS] COMMAND [ARGS]...

  Room Impulse Response Measuring and Evaluation

Options:
  --loglevel TEXT  Set the logging level {[INFO], DEBUG, WARNING, ERROR,
                   CRITICAL}.
  --help           Show this message and exit.

Commands:
  analyze   Analyze measurement results.
  convolve  Convolve and listen to measurement results.
  devices   List available sound devices.
  init      Initialize rirmeasuring project in path.
  new       Create new rirmeasuring project folder and initialize it.
  run       Run RIR MEasurement

```


```shell
$ rirme new --help
Usage: rirme new [OPTIONS] NAME

  Create new rirmeasuring project folder and initialize it.

Options:
  -p, --path TEXT  Path where the rirmeasuring project folder will be created.
  --help           Show this message and exit.

```


```shell
$ rirme init --help
Usage: rirme init [OPTIONS] NAME

  Initialize rirmeasuring project in path.

Options:
  -p, --path TEXT  Path where the config and project files are placed.
  --help           Show this message and exit.

```


```shell
$ rirme devices
    ...
   9 Fireface UCX (23671730): USB Audio (hw:2,0), ALSA (18 in, 18 out)
  10 sysdefault, ALSA (128 in, 128 out)
  11 front, ALSA (0 in, 2 out)
    ...
  16 pulse, ALSA (32 in, 32 out)
  17 dmix, ALSA (0 in, 2 out)
* 18 default, ALSA (32 in, 32 out)
    ...
```
```shell
$ rirme analyze --help
Usage: rirme analyze [OPTIONS] [NAME]

  Analyze measurement results.  NAME is either a explicit filename or a part
  of the filename(s) that will be globbed.

Options:
  -s, --save TEXT
  -d, --dpi INTEGER
  -p, --path TEXT
  --show
  --help             Show this message and exit.
```

```shell
$ rirme convolve --help
Usage: rirme convolve [OPTIONS] [NAME]

  Convolve and listen to measurement results.

Options:
  -i, --input TEXT
  -s, --save TEXT
  -p, --path TEXT
  -q, --quiet
  -l, --level INTEGER
  --help               Show this message and exit.



