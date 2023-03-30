### ORIGEN Library Manager (OLM)

Structure from 

https://realpython.com/python-application-layouts/#command-line-application-layouts

```
❯ tree .
.
├── LICENSE
├── README.md
├── data
│   ├── ge10x10-8.h5
│   ├── mox_w17x17.h5
│   ├── vver440.h5
│   └── w17x17.h5
├── debug.ipynb
├── docs
│   └── index.rst
├── scale
│   └── olm
│       ├── Archive.py
│       ├── __init__.py
│       ├── __main__.py
│       ├── check
│       │   ├── CheckInfo.py
│       │   ├── __init__.py
│       │   ├── check.py
│       │   └── helpers.py
│       ├── link
│       │   ├── __init__.py
│       │   ├── helpers.py
│       │   └── link.py
│       └── olm.py
└── tests
    ├── check
    │   ├── check_tests.py
    │   └── helpers_tests.py
    └── link
        ├── helpers_tests.py
        └── link_tests.py

9 directories, 23 files
```

In order to create data/w17x17.h5 do this:
```
obiwan convert -alias=w17x17 -format=hdf5 ${DATA}/arpdata.txt
```
