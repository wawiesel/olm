Step-by-step
------------

The quickstart performed all the stages of library creation in one command. Here we
will go step-by-step and explain in more detail. If you have already created the
reactor library in the quick start.


Initializing a new library
~~~~~~~~~~~~~~~~~~~~~~~~~~

Initialize a new reactor library directory as follows.

.. code:: console

    $ olm init --variant=uox_quick
    
    $ tree uox_quick

    uox_quick
    ├── config.olm.json
    ├── model.jt.inp
    └── report.jt.rst
    
    1 directory, 3 files

OLM uses templates for SCALE input files which are populated with data defined in special
JSON files with extension :code:`.olm.json`. The :code:`config.olm.json` contains the configuration
data that defines the reactor library, such as the maximum burnup to consider or the enrichment range.
The :code:`template.inp` contains a basic TRITON input template from which to generate all the
permutations needed.

Inspect the files
~~~~~~~~~~~~~~~~~

The :code:`model.jt.inp` file created contains 
`Jinja <https://jinja.palletsprojects.com/en/3.1.x/templates/#synopsis>`_ 
template directives. This SCALE input template is "expanded" into a proper SCALE
input file using data contained in the :code:`config.olm.json`.

.. code:: scale

    =t-depl parm=(bonami,addnux={{static.addnux}})
    pincell model
    {{static.xslib}}

    read composition
    '
    ' fuel
      uo2   10 den={{comp.fuel.density}} 1
               900
               92234 {{comp.fuel.uo2.iso.u234}}
               92235 {{comp.fuel.uo2.iso.u235}}
               92236 {{comp.fuel.uo2.iso.u236}}
               92238 {{comp.fuel.uo2.iso.u238}} end
    '
    ' clad
      zirc4 20 1 622 end
    '
    ' coolant
      h2o   30 den={{state.coolant_density}} 1.000000 575 end
      boron 30 den={{state.coolant_density}} {{state.boron_ppm*1e-6}} 575 end
    '
    end composition

    read celldata
      latticecell squarepitch
        pitch={{static.pitch}} 30
        fuelr={{static.fuelr}} 10
        cladr={{static.cladr}} 20 end
    end celldata

    read depletion
      10
    end depletion

    read burndata
      {%- for pb in time.burndata %}
      power={{pb.power}} burn={{pb.burn}} down=0 end
      {%- endfor %}
    end burndata

    read model

    read materials
      mix=10 pn=1 com="fuel" end
      mix=20 pn=1 com="clad" end
      mix=30 pn=2 com="coolant" end
    end materials

    read geom
      global unit 1
        cylinder 10 {{static.fuelr}}
        cylinder 20 {{static.cladr}}
        cuboid   30 4p{{static.pitch/2.0}}
        media 10 1 10
        media 20 1 20 -10
        media 30 1 30 -20
      boundary 30 3 3
    end geom

    read bounds
      all=refl
    end bounds

    end model
    end


Generating inputs
~~~~~~~~~~~~~~~~~

.. code:: console

    $ olm create --generate uox_quick/config.olm.json
    
    $ tree uox_quick/_work
    
    uox_quick/_work
    ├── env.olm.json
    ├── generate.olm.json
    └── perms
        ├── 9a0c0e1d2b0e77171b515f9a3cb5d239
        │   ├── data.olm.json
        │   └── model_b5d239.inp
        ├── 9b0ce11728be1c18a884dd6260bcf24a
        │   ├── data.olm.json
        │   └── model_bcf24a.inp
        ├── 9c0c9dc8a956c93a4dce918e77924408
        │   ├── data.olm.json
        │   └── model_924408.inp
        ├── 9c0caee0bd2ebe529726fc5692e86612
        │   ├── data.olm.json
        │   └── model_e86612.inp
        ├── 9d0c007d60fedfb6064c26e03e755bd3
        │   ├── data.olm.json
        │   └── model_755bd3.inp
        ├── 9d0c0a5ecd793d6c0a371838fd722763
        │   ├── data.olm.json
        │   └── model_722763.inp
        ├── a50cc0c173d75f402cc298050465473a
        │   ├── data.olm.json
        │   └── model_65473a.inp
        ├── a60cae65b0efc59e1a7613bb4247a46d
        │   ├── data.olm.json
        │   └── model_47a46d.inp
        ├── a70c8e9c503202040438e3be436689cd
        │   ├── data.olm.json
        │   └── model_6689cd.inp
        ├── a70cb3ff2656d59cc698438e0f578dc7
        │   ├── data.olm.json
        │   └── model_578dc7.inp
        ├── a80cdb85fc2adb1cd485a16c8d939887
        │   ├── data.olm.json
        │   └── model_939887.inp
        └── a80cf3a2e9535ae102007cb266b84f9a
            ├── data.olm.json
            └── model_b84f9a.inp
    
    14 directories, 26 files

You will notice that all calculations happen in a work directory, which my default is
next to the :code:`config.olm.json` file as :code:`_work`. It can be changed by setting the environment
variable :code:`SCALE_OLM_WORK`.

Running the calculations
~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: This example is set up so that the entire process runs in less than a minute on 3 cores.

The :code:`olm create --run` requires access to the SCALE runtime, :code:`scalerte`. Operations in
the :code:`--assemble` stage require access to :code:`OBIWAN` which ships with SCALE. Setting the
environment variable :code:`SCALE_DIR` will find them both.

.. code:: console

    $ export SCALE_DIR=/Applications/SCALE-6.3.2.app/Contents/Resources

    $ olm create -j6 uox_quick/config.olm.json

With that set, you can create an ORIGEN reactor library with :code:`olm create`
and the path to the configuration file, :code:`config.olm.json`.

This is the part of the reactor library generation that can take a very long time.
In our quick examples, we change the TRITON runtime parameters to be very fast and
approximate, but typical production calculations of this type can take CPU-days. For
this reason, you may want to run these calculations outside of OLM, for example on
a cluster. The example uses a simple Makefile approach to run the SCALE inputs which
does allow for some parallelism and recovery from keyboard interrupts. The Makefile
generated by OLM has some limited capability to not rerun calculations if they
completed successfully.

OLM has a modular approach to each stage, defined inside the :code:`config.olm.json`. For
the run stage of this example, the following block determines how the run is performed.

.. code::json

    "run": {
        "_type": "scale.olm.run:makefile",
        "nprocs": 3,
        "dry_run": false
    }

The :code:`_type` key specifies a function to run, in this case :code:`makefile` inside the module
`scale.olm.run`. This particular run function allows specification of the number of
processes to use and whether it is a dry run or not. Change the "dry_run" from false
to true before you type.

You should see screen output as the calculation proceeds. All calculations happen in a
working directory which is by default :code:`_work` in the same parent directory as
the configuration file. The :code:`-j6` specifies to use 6 processes to generate the
library.

.. code:: console

    $ tree -L 2 uox_quick/_work

	uox_quick/_work
	├── arpdata.arc.h5
	├── arpdata.txt
	├── arplibs
	│   ├── uox_quick_e0050w0700.h5
	│   ├── uox_quick_e0050w0700.ii.json
	│   ├── uox_quick_e0050w0720.h5
	│   ├── uox_quick_e0050w0720.ii.json
	│   ├── uox_quick_e0050w0740.h5
	│   ├── uox_quick_e0050w0740.ii.json
	│   ├── uox_quick_e0300w0700.h5
	│   ├── uox_quick_e0300w0700.ii.json
	│   ├── uox_quick_e0300w0720.h5
	│   ├── uox_quick_e0300w0720.ii.json
	│   ├── uox_quick_e0300w0740.h5
	│   ├── uox_quick_e0300w0740.ii.json
	│   ├── uox_quick_e0500w0700.h5
	│   ├── uox_quick_e0500w0700.ii.json
	│   ├── uox_quick_e0500w0720.h5
	│   ├── uox_quick_e0500w0720.ii.json
	│   ├── uox_quick_e0500w0740.h5
	│   ├── uox_quick_e0500w0740.ii.json
	│   ├── uox_quick_e0700w0700.h5
	│   ├── uox_quick_e0700w0700.ii.json
	│   ├── uox_quick_e0700w0720.h5
	│   ├── uox_quick_e0700w0720.ii.json
	│   ├── uox_quick_e0700w0740.h5
	│   └── uox_quick_e0700w0740.ii.json
	├── assemble.olm.json
	├── check
	│   └── loc
	├── check.olm.json
	├── env.olm.json
	├── generate.olm.json
	├── perms
	│   ├── 9a0c0e1d2b0e77171b515f9a3cb5d239
	│   ├── 9b0ce11728be1c18a884dd6260bcf24a
	│   ├── 9c0c9dc8a956c93a4dce918e77924408
	│   ├── 9c0caee0bd2ebe529726fc5692e86612
	│   ├── 9d0c007d60fedfb6064c26e03e755bd3
	│   ├── 9d0c0a5ecd793d6c0a371838fd722763
	│   ├── Makefile
	│   ├── a50cc0c173d75f402cc298050465473a
	│   ├── a60cae65b0efc59e1a7613bb4247a46d
	│   ├── a70c8e9c503202040438e3be436689cd
	│   ├── a70cb3ff2656d59cc698438e0f578dc7
	│   ├── a80cdb85fc2adb1cd485a16c8d939887
	│   └── a80cf3a2e9535ae102007cb266b84f9a
	├── report.olm.json
	├── run.olm.json
	├── uox_quick.pdf
	└── uox_quick.rst

	17 directories, 35 files


.. note:: To see more output you can set the environment variable :code:`SCALE_LOG_LEVEL`,
          where 10 is DEBUG, 20 is INFO, 30 is WARNING, 40 is ERROR only. See
          `structlog levels <https://docs.python.org/3/library/logging.html#logging-levels>`_
          for details.


Using the library in ORIGAMI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`olm link` command can take a specific path with :code:`-p`, or it can read
from a list of paths in the :code:`SCALE_OLM_PATH` variable. This variable is just like
the Linux :code:`PATH` variable or :code:`PYTHON_PATH` used to find modules. It is a
colon-separated (:) list of paths, searched first to last.

The recommended way to use this library is to first set the path variable. 

.. code:: console

    $ export SCALE_OLM_PATH=$PWD/uox_quick/_work
    
Copy/paste the input below into a file, :code:`origami.inp`.

.. literalinclude:: origami.inp
	:language: scale

Run the input with SCALE.

.. code:: console

    $ export SCALE_OLM_PATH=$PWD/uox_quick/_work
    
    $ $SCALE_DIR/bin/scalerte -m origami.inp
    





