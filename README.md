### ORIGEN Library Manager (OLM)

Structure from 

https://realpython.com/python-application-layouts/#command-line-application-layouts

```
.
├── LICENSE
├── README.md
├── bin
│   └── olm.py
├── data
│   ├── x1.arc.h5
│   └── x2.arc.h5
├── docs
│   └── index.rst
├── olm
│   ├── __init__.py
│   ├── check
│   │   ├── __init__.py
│   │   ├── check.py
│   │   └── helpers.py
│   ├── link
│   │   ├── __init__.py
│   │   ├── helpers.py
│   │   └── link.py
│   └── runner.py
└── tests
    ├── check
    │   ├── check_tests.py
    │   └── helpers_tests.py
    └── link
        ├── helpers_tests.py
        └── link_tests.py

9 directories, 18 files
```

In order to create data/w17x17.h5 do this:
```
obiwan convert -alias=w17x17 -format=hdf5 ${DATA}/arpdata.txt
```
