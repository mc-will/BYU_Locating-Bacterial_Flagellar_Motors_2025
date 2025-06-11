from setuptools import setup, find_packages

setup(
    name='source',
    description="Code for flagellar motor project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
