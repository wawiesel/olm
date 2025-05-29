# 🚀 ORIGEN Library Manager (OLM)

> **Build, manage, and validate ORIGEN reactor libraries with ease**

[![Documentation Status](https://readthedocs.org/projects/scale-olm/badge/?version=v0.14.2)](https://scale-olm.readthedocs.io/en/v0.14.2)
[![PyPI version](https://badge.fury.io/py/scale-olm.svg)](https://badge.fury.io/py/scale-olm)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD-green.svg)](LICENSE)

**OLM** streamlines the complex process of creating, validating, and managing [SCALE/ORIGEN](https://scale.ornl.gov) reactor libraries for nuclide inventory calculations. Say goodbye to manual library management! 🎯

## ⚡ Quick Start

Get up and running in 5-10 minutes, starting with a clean slate and ending with a simple UOX ORIGEN reactor library:

```bash
# Initialize a configuration file for the uox_quick variant
olm init --variant uox_quick

# Create an ORIGEN library with parallel processing
olm create -j6 uox_quick/config.olm.json

# Open the generated report
open uox_quick/_work/uox_quick.pdf
```

### 🎯 **Using Your New Library in SCALE**

Once created, you can use your library in ORIGAMI calculations:

```bash
# Allow the local library to be found by olm link
export SCALE_OLM_PATH=$PWD/uox_quick/_work
```

Create an ORIGAMI input file (`origami.inp`):
```
=shell
olm link uox_quick
end

=origami
libs=[ uox_quick ]

fuelcomp{
    uox(fuel){ enrich=4.95 }
    mix(1){ comps[fuel=100] }
}

modz = [ 0.74 ]
pz = [ 1.0 ]

hist[
  cycle{ power=40 burn=1000 nlib=10 }
]
end
```

Then run your calculation:
```bash
$SCALE_DIR/bin/scalerte -m origami.inp
```

That's it! You now have a verified ORIGEN library ready for spent fuel inventory generation. ✨

> **📋 Requirements**: Generating ORIGEN libraries requires SCALE 6.3.2+. Using UOX libraries requires SCALE 6.2.4+. MOX libraries require SCALE 7.0+.

> **💡 Tip**: Set `SCALE_LOG_LEVEL=30` to show only warnings and errors during library creation.

## 🎯 What OLM Does

| Challenge | OLM Solution |
|-----------|--------------|
| 📊 **Complex Library Creation** | One-command library generation from reactor parameters |
| 🔍 **Quality Validation** | Automated quality checks with numerical scoring |
| 📦 **Library Management** | Install, link, and organize libraries effortlessly |
| 🔧 **Workflow Automation** | Parallel processing and dependency management |
| 📈 **Reproducibility** | JSON-based configuration for version control |


## 🚀 Installation Options

### 📦 PyPI (Recommended)
```bash
pip install scale-olm
```

### Quick Development Setup
```bash
git clone https://github.com/wawiesel/olm
cd olm
source dev.sh          # Automatic environment setup
pre-commit install     # Enable code formatting
pytest -n 6 .          # Run tests in parallel
```

## 📚 Documentation & Resources

- 📖 **[Official Documentation](https://scale-olm.readthedocs.io/en/stable)** - Complete user guide
- 🐙 **[GitHub Repository](https://github.com/wawiesel/olm)** - Source code and issues
- 🧪 **[Examples](examples/)** - Ready-to-run reactor configurations
- 📓 **[Notebooks](notebooks/)** - Interactive tutorials and debugging guides

## What We Use
- 🐍 **Python 3.9+** with modern async/await patterns
- 🖱️ **[Click](https://click.palletsprojects.com/)** for beautiful CLI interfaces  
- ⚫ **[Black](https://black.readthedocs.io/)** for consistent code formatting
- 🧪 **[Pytest](https://pytest.org/)** with parallel test execution
- 📝 **[Sphinx](https://www.sphinx-doc.org/)** for documentation

### Development Guidelines
- 📝 Follow [conventional commit messages](https://cbea.ms/git-commit/)
- 🔖 Use [semantic versioning](https://semver.org/) (`bumpversion patch|minor|major`)
- 🧪 Write tests for new features
- 📚 Document public APIs with docstrings

---

## 🏢 Repository Locations

- **Primary**: [GitHub](https://github.com/wawiesel/olm) (main development)
- **Mirror**: [ORNL GitLab](https://code.ornl.gov/scale/code/olm) (read-only)

