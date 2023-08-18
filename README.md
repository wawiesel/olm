### ORIGEN Library Manager (OLM)

Here's the current repo structure. The main code is in `scale/olm/__main__.py`.

```
tree -I venv -I _build
.
├── LICENSE
├── README.md
├── data
│   ├── ge10x10-8.h5
│   ├── mox_w17x17.h5
│   ├── vver440.h5
│   └── w17x17.h5
├── docs
│   └── index.rst
├── examples
│   └── w17x17
│       ├── config-olm.json
│       └── model.inp
├── notebooks
│   └── debug.ipynb
├── requirements-dev.txt
├── scale
│   └── olm
│       ├── __init__.py
│       ├── __main__.py
│       ├── build.py
│       ├── check.py
│       ├── common.py
│       ├── generate.py
│       ├── link.py
│       ├── report.py
│       └── run.py
├── setup.py
└── testing
    ├── check_test.py
    ├── common_test.py
    └── generate_test.py

9 directories, 30 files
```

In order to create data/w17x17.h5 do this:
```
obiwan convert -alias=w17x17 -format=hdf5 ${DATA}/arpdata.txt
```

### Enable virtual environment

```
virtualenv venv
. venv/bin/activate
which python
```

### Install requirements

Enable the virtual environment, then run this command to install dependencies.

```
pip install -r requirements-dev.txt
```

Regenerate the requirements file like this.

```
pip freeze | grep -v '^\-e'>requirements-dev.txt
```

### Local install for testing

```
pip install --editable .
olm
which olm
```

### Click for CLI

We use the Click python library https://click.palletsprojects.com/en/8.1.x/
for command line.

Here's a nice video.

https://www.youtube.com/watch?v=kNke39OZ2k0


### Run a check

Here's how you run a check from the command line.

```
olm check -s '{".type": "GridGradient" }' data/w17x17.h5
```

### Use an OLM configuration file

The configuration file contains all the information to move through all the stages
of library generation, checking, and reporting. It requires knowing the SCALE 
install you want to use.

```
export SCALE=/Applications/SCALE-6.3.0.app/Contents/Resources
```

The stages are:

1. **generate** inputs
2. **run** inputs
3. **build** ORIGEN library from outputs
4. **check** ORIGEN library
5. **report** on aspects of the library

See `olm do` help screen for details. For example, this command 
generates the different file permutations.

```
olm do --generate examples/w17x17/config-olm.json
```

This command runs them with 3 local processors.

```
olm do --run examples/w17x17/config-olm.json --nprocs 3
```

Multiple stages can be run at once.

```
olm do --generate --run --build examples/w17x17/config-olm.json
```

All stages can be run with `--all`.

```
cd examples/quick2
olm do --all config-olm.json
```

### Run the unit tests

Locally for unit tests we use the pytest framework under the `testing` directory.
All tests can be run simply like this from the root directory.

```
pytest .
```

### Local formatting on commit

Use the black formatter, https://medium.com/gousto-engineering-techbrunch/automate-python-code-formatting-with-black-and-pre-commit-ebc69dcc5e03