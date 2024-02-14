import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scale-olm",
    author="William Wieselquist",
    author_email="ww5@ornl.gov",
    description="ORIGEN Library Manager",
    keywords="SCALE, ORIGEN, Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://code.ornl.gov/scale/code/olm",
    project_urls={
        "Documentation": "https://code.ornl.gov/scale/code/olm",
        "Bug Reports": "https://code.ornl.gov/scale/code/olm/issues",
        "Source Code": "https://code.ornl.gov/scale/code/olm",
        # 'Funding': '',
        # 'Say Thanks!': '',
    },
    package_dir={"": "scale"},
    packages=setuptools.find_packages(where="scale-olm"),
    classifiers=[
        # see https://pypi.org/classifiers/
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Nuclear Physics",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: BSD-2",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=["matplotlib", "scipy", "numpy", "pytest", "click", "pydantic"],
    extras_require={
        "dev": ["check-manifest"],
        # 'test': ['coverage'],
    },
)
