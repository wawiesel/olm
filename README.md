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

That's it! You now have a validated ORIGEN library ready for reactor simulations. ✨

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

## 🌟 Key Features

### 🏗️ **Automated Library Building**
Transform reactor specifications into production-ready ORIGEN libraries
```bash
olm create --generate --run --assemble --check config.olm.json
```

### 🎯 **Smart Initialization**  
Choose from pre-built reactor variants or create custom configurations
```bash
olm init --list  # See available reactor types
olm init --variant mox_quick  # Start with MOX fuel template
```

### 🔍 **Quality Assurance**
Validate libraries with comprehensive numerical checks
```bash
olm check -s '{"_type": "GridGradient", "eps0": 1e-6}' my_library.arc.h5
```

### 📦 **Library Management**
Install and link libraries for seamless SCALE integration
```bash
olm install --overwrite my_project/_work
olm link uox_quick  # Ready to use in SCALE inputs!
```

## 🚀 Installation Options

### 📦 PyPI (Recommended)
```bash
pip install scale-olm
```

### 🔧 Development Setup
```bash
git clone https://github.com/wawiesel/olm
cd olm
source dev.sh  # Sets up everything automatically!
```

## 📋 Typical Workflow

Here's how OLM fits into your reactor analysis workflow:

```mermaid
graph LR
    A[Define Reactor] --> B[olm init]
    B --> C[Configure Parameters]
    C --> D[olm create]
    D --> E[olm check]
    E --> F[olm install]
    F --> G[Use in SCALE]
```

### Step-by-Step Example

```bash
# 1. Start a new UOX reactor library project
olm init --variant uox_quick
cd uox_quick

# 2. Build the library with parallel processing
olm create -j6 config.olm.json

# 3. Validate library quality
olm check ../data/w17x17.arc.h5

# 4. Install for use in SCALE
olm install --overwrite _work
export SCALE_OLM_PATH=$HOME/.olm

# 5. Use in your SCALE input files
# =origami
# lib=[ uox_quick ]
# ...
# end
```

## 🛠️ Available Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `init` | Start new project | `olm init --variant mox_quick` |
| `create` | Build library | `olm create -j6 config.olm.json` |
| `check` | Validate quality | `olm check my_library.arc.h5` |
| `install` | Deploy library | `olm install --overwrite _work` |
| `link` | Connect to SCALE | `olm link my_library` |
| `schema` | View configuration | `olm schema scale.olm.generate.comp:uo2_simple` |

## 📚 Documentation & Resources

- 📖 **[Official Documentation](https://scale-olm.readthedocs.io/en/stable)** - Complete user guide
- 🐙 **[GitHub Repository](https://github.com/wawiesel/olm)** - Source code and issues
- 🧪 **[Examples](examples/)** - Ready-to-run reactor configurations
- 📓 **[Notebooks](notebooks/)** - Interactive tutorials and debugging guides

## 🤝 Contributing

We ❤️ contributions! Whether you're fixing bugs, adding features, or improving documentation:

### Quick Development Setup
```bash
git clone https://github.com/wawiesel/olm
cd olm
source dev.sh          # Automatic environment setup
pre-commit install     # Enable code formatting
pytest -n 6 .          # Run tests in parallel
```

### What We Use
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

## 📊 Project Status

- **Latest Stable**: [v0.14.2](https://scale-olm.readthedocs.io/en/stable) 
- **Python Support**: 3.9, 3.10, 3.11
- **Development Status**: Active
- **License**: BSD-3-Clause

---

## 🏢 Repository Locations

- **Primary**: [GitHub](https://github.com/wawiesel/olm) (main development)
- **Mirror**: [ORNL GitLab](https://code.ornl.gov/scale/code/olm) (read-only)

---

**Ready to streamline your ORIGEN workflows?** [Get started now!](#-quick-start) 🚀
