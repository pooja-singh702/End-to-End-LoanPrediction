## responsible for creating my ml project as a package.
## then u can install it later or use it .

## it will help to find all packages used in the dir created
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT= "-e ."
def get_requirements(file_path:str)->List[str]:
    ## this func returns list of requiremntsto install 
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)





## metadata info about entire project
setup(
    name="End to End Loan Prediction", 
    version = "0.0.1",
    author = "Pooja Singh",
    author_email = "mail.poojasingh11@gmail.com",
    packages=find_packages(),
    # instead listing manually all library using funct to use requirement.txt all names as list
    install_requires=get_requirements("requirements.txt"))