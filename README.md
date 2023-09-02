## What is OLM?

OLM is the ORIGEN Library Manager, a command line utility that streamlines
aspects of using the ORIGEN library to solve nuclide inventory generation problems.

If you would like to learn how to use OLM, see the (online manual)[scale-olm.readthedocs.io].
If you would like to learn how to develop OLM, continue reading.

### Enable virtual environment

```
virtualenv venv
. venv/bin/activate
which python
```

If you get an error about missing `virtualenv`, you may need to run this
`pip install virtualenv`.

### Install requirements

After enabling the virtual environment, run this command to install dependencies.

```
pip install -r requirements-dev.txt
```

NOTE: if you need to regenerate the requirements file after adding dependencies.
```
pip freeze | grep -v '^\-e'>requirements-dev.txt
```

### Enable a local install for testing

This command will enable any changes you make to instantly propagate to the executable
you can run just with `olm`.

```
pip install --editable .
olm
which olm
```

### Notebooks

There are notebooks contained in `notebooks` which may be helpful for debugging or
understanding how something is working. You may need to install your virtual environment
kernel for the notebooks to work.

```
ipython kernel install --name "venv" --user
```

Now, you can select the created kernel "venv" when you start Jupyter notebook or lab.

## Notes about development

### Click for CLI

We use the Click python library https://click.palletsprojects.com/en/8.1.x/
for command line.

Here's a nice video.

https://www.youtube.com/watch?v=kNke39OZ2k0


### Pytest for unit tests

Locally for unit tests we use the pytest framework under the `testing` directory.
All tests can be run simply like this from the root directory.

```
pytest .
```

### Black for commit formatting

The first time you do work on a clone, do this.
```
pre-commit install
```

This will use the black formatter,
https://medium.com/gousto-engineering-techbrunch/automate-python-code-formatting-with-black-and-pre-commit-ebc69dcc5e03

### Docstrings and Doctest

Our goal is to have each function, module, and class with standard docstrings and
a few doctests. Doctests should be run like this (not `python -m doctest -v scale/olm/core.py`)
due to usage of some reusable global class instances that are only enabled when the module
itself is `__main__`.

```
python scale/olm/core.py -v
```
