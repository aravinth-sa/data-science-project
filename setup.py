from setuptools import find_packages, setup

def get_requirements(file_path):
    requirement = []
    with open(file_path, 'r') as file:
        requirement =  file.readlines()
        requirement = [pkg.replace("\n","") for pkg in requirement]
        if '-e .' in requirement:
            requirement.remove('-e .')
    return requirement

setup(
    name='data-science-project',
    version="0.1.0",
    author='Aravinth',
    author_email='ars.aravinth@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirement.txt'),
)

