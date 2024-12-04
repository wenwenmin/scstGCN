from setuptools import Command, find_packages, setup

with open("README.rst", "r", encoding="utf-8") as f:
    __long_description__ = f.read()

setup(
    name = "STAGE",
    version = "1.0.1",
    description = "High-density generation of spatial transcriptomics with STAGE",
    url = "https://github.com/zhanglabtools/STAGE",
    author = "Shang Li",
    author_email = "lishang@amss.ac.cn",
    license = "MIT",
    packages = ['STAGE'],
    install_requires = ["requests",],
    zip_safe = False,
    include_package_data = True,
    long_description = __long_description__
)
