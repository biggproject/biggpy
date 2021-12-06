# AI Toolbox
The Building Information aGGregation, harmonisation and analytics (BIGG) platform is a EU-funded project to aims at 
demonstrating the application of big data technologies and data analytics techniques for the complete buildings 
life-cycle of more than 4000 buildings in 6 large-scale pilot test-beds. 
This repository contains the python library AI Toolbox, which will provide all the AI tools necessary to build a 
machine learning pipeline in the context of WP5 and related business cases and use cases.
For the language-agnostic documentation, please refer to: 
https://github.com/biggproject/biggdocs

## Installation
It is highly recommended setting up a virtual environment before using this package:

```bash
python3 -m venv ai_toolbox_venv/
source ai_toolbox_venv/bin/activate
```

Once the venv is activated, **cd to the ai_toolbox package dir**. For example:
```bash
cd ~/projects/ai_toolbox
```
run inside the venv:

#### Development
```bash
pip3 install -e .
```

This command should be executed by developers. It will install this package in "editable" mode using the script setup.py
and all its dependencies. This implies that any changes to the original package would reflect directly in your 
environment.

#### Production 
```bash
pip3 install .
```

## Requirements
Requires-Python: ">=3.6, <3.9"


## Examples
Some examples of usage are available as Jupyter notebooks in the directory "notebooks":
https://github.com/biggproject/biggpy/tree/main/ai_toolbox/notebooks

## Tests

### Testing in local (or virtual) environment
The directory "tests" contains a test file for each module of the AI toolbox. Unit tests make sure that
each module behaves as intended and that future code changes will not break the current code.
Once the package is installed, to run tests using the automatic discovery mechanism, "cd" to the directory
of the "ai_toolbox".
For example:
```bash
cd ~/projects/ai_toolbox
```
Then, from inside the directory, type:
```bash
python3 -m unittest -v
```

Note: To run successfully the application and tests the package **must be installed** (normal or editable mode) 
otherwise it will throw import errors. Please follow the instructions in the "Installation" section.

### Testing in Multiple Environments
The library uses tox to automate testing in multiple environments. 
To install tox on your system run:
```bash
pip3 install tox
```
For further information check out: https://tox.wiki/en/latest/index.html .
Tox will read the configuration file ```tox.ini```, 
create a virtual environment for each environment specified in the section ```env```, item ```envlist```, and run the
tests for each of them. In the end, tox will provide a final report with all the commands executed.
To run tox, **cd to the ai_toolbox package dir**. For example:
```bash
cd ~/projects/ai_toolbox
```
Then, run:
```bash
tox
```

## Credits
* Gerard Mor, gmor@cimne.upc.edu
* Manu Lahariya, Manu.Lahariya@ugent.be
* Riccardo De Vivo, Riccardo.DeVivo@energis.cloud
* Théa Gutmacher, Thea.Gutmacher@inetum-realdolmen.world

## Copyright

BIGG copyright here, if any