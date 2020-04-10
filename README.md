# csci508-final
Final Project for Advanced Computer Vision - Recycleable Classifier 


## Installation and Usage
Installing this project can be done in a variety of ways but the suggested method relies on using a virtual environment 
provided through [Poetry](https://python-poetry.org/docs/).

It is also recommended that you use pyenv to manage your variety of python environments. Pyenv can be installed by 
executing:
```
$ curl https://pyenv.run | bash
```
Additional help for setting up pyenv can be found [here](https://realpython.com/intro-to-pyenv/)

Once you have pyenv installed add in your preferred version of Python 3.6, or simply default to Python 3.6.9 (the latest 
version before the release of 3.7) and set pyenv to use this version for this project. 

With the project cloned to you local system, change directory into the project's top level. In order to install all of 
the project dependencies to the virtual environment execute:
```
$ poetry install
```

And in order to run this project simply execute:
```
$ poetry run python csci508_final/main.py
```

## Data Collection
In order to compare the performance of various network architectures and methods on our data we are collecting all of 
our results in a shared google sheets project which can be found and modified 
[here](https://drive.google.com/open?id=1LFFuCYt-rlyDO3pLFGtBgJ-dwgkvgq0lGIOQKgtxyao). Note that this is a fully mutable 
sheet as the number and variety of possible hyper-parameters varies with architectures. This may become more 
constrained at some point but as things evolve this flexibility is more critical than safety. 
