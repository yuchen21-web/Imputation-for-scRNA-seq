from distutils.core import setup

setup(
    # package_dir={'':'CLIMP'},
    packages=['model'],
    name="CLImpute",
    version="1.0",
    description="Contrastive Learning-based method for single cell imputation",
    author="Yuchen Shi",
    author_email="yuchen@hdu.edu.cn",
    py_modules=["CLImputeUtils"]
    )
