## ORIGEN Library Manager (OLM)

[![Documentation Status](https://readthedocs.org/projects/scale-olm/badge/?version=v0.12.3)](https://scale-olm.readthedocs.io/en/v0.12.3)

The latest stable version is [v0.12.3](https://scale-olm.readthedocs.io/en/stable).

OLM is a command-line utility that streamlines aspects of using the 
[SCALE/ORIGEN](https://scale.ornl.gov) library to solve nuclide inventory generation problems.

To install, use `pip`.

```console
pip install scale-olm
```

## Locations

The main development repository is hosted on [GitHub](https://github.com/wawiesel/olm) 
with a read-only mirror on the ORNL-hosted [GitLab](https://code.ornl.gov/scale/code/olm).

## Developing

The script `dev.sh` is provided to initialize the development environment.

```console
$ git clone https://github.com/wawiesel/olm
$ cd olm
$ source dev.sh
```

This is all you should need to do. The following sections explain in more detail 
what happens when you run `dev.sh`.

## Developer details

This section contains additional details on developing OLM.

### Enable virtual environment

```console
$ virtualenv venv
$ . venv/bin/activate
$ which python
```

If you get an error about missing `virtualenv`, you may need to install it.

```console
$ pip install virtualenv
```

### Install requirements

After enabling the virtual environment, run this command to install dependencies.

```console
$ pip install -r requirements.txt
```

NOTE: if you need to regenerate the requirements file after adding dependencies.
```console
$ pip freeze | grep -v '^\-e'>requirements.txt
```

### Enable a local install for testing

This command will enable any changes you make to instantly propagate to the executable
you can run just with `olm`.

```console
$ pip install --editable .
$ olm
$ which olm
```

### Creating docs

With the development environment installed, the docs may be created within the
`docs` directory. With the following commands.

```console
$ cd docs
$ make html
$ open build/html/index.html
```

Alternatively the PDF docs may be generated using the `make latexpdf` command. Note
that the HTML docs are intended as the main documentation.

The following greatly simplifies iterating on documentation. Run this command
and open your browser to http://localhost:8000.

```console
sphinx-autobuild docs/source/ docs/build/html/
```



### Notebooks

There are notebooks contained in `notebooks` which may be helpful for debugging or
understanding how something is working. You may need to install your virtual environment
kernel for the notebooks to work. You should use the local `venv` kernel instead of
your default Python kernel so you have all the local packages at the correct versions.

```console
$ ipython kernel install --name "venv" --user
```

Now, you can select the created kernel "venv" when you start Jupyter notebook or lab.

## Notes about development

### Click for CLI

We use the [Click python library](https://click.palletsprojects.com/en/8.1.x)
for command line. Here's a nice [video about click](https://www.youtube.com/watch?v=kNke39OZ2k0).

### Commit messages

Follow these [guidelines](https://cbea.ms/git-commit/) for commit messages.

### Version updates

OLM uses [semantic versioning](https://semver.org/). You should commit the 
relevant code with the usual description commit message. 

Then run 

- `bumpversion patch` if you are fixing a bug
- `bumpversion minor` if you are adding a new feature
- `bumpversion major` if you are breaking backwards compatibility

When you push you need to `git push --tags` or configure your repo to always push tags:

```
#.git/config
[remote "origin"]
    push = +refs/heads/*:refs/heads/*
    push = +refs/tags/*:refs/tags/*
```

### Pytest for unit tests

Locally for unit tests we use the pytest framework under the `testing` directory.
All tests can be run simply like this from the root directory. Not we are using the
`pytest-xdist` extension which allows parallel testing.

```console
$ pytest -n 6 .
```

### Black for commit formatting

The first time you do work on a clone, do this.

```console
$ pre-commit install
```

This will use the [Black formatter](https://medium.com/gousto-engineering-techbrunch/automate-python-code-formatting-with-black-and-pre-commit-ebc69dcc5e03).


### Docstrings and Doctest

Our goal is to have each function, module, and class with standard docstrings and
a few doctests. You can run verbose tests on a specific module as follows.

```console
$ pytest -v scale/olm/core.py
```
