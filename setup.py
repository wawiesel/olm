from setuptools import setup, find_namespace_packages

setup(
    name="scale-olm",
    version="0.1",
    packages=find_namespace_packages(include=["scale.*"]),
    include_package_data=True,
    install_requires=[
        "Click",
    ],
    entry_points={"console_scripts": ["olm=scale.olm.__main__:cli"]},
)
