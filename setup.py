# setup.py is to used to convert our project into a package that can be installed and used as like the other packages when we run this file then it will search __init__.py file containing folders and those file which contain it will be used a made as a packages



from setuptools import find_packages, setup
from typing import List



def get_requirements(file_path:str)->List[str]:
    '''
        this function will return the list of requirements
    '''
    hypen_E_dot = "-e ."
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        # if hypen_E_dot in requirements:
        #     requirements.remove(hypen_E_dot)
    return requirements





setup(
    name = "ml project",
    version = "0.0.1",
    author = "Abhi",
    author_email = "abhiabhishek22365@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)