# csci508-final
Final Project for Advanced Computer Vision - Recycleable Classifier 


## Installation and Usage
Installing this project can be done in a variety of ways but the suggested method relies on using a virtual environment provided through [Poetry](https://python-poetry.org/docs/).

It is also recommended that you use pyenv to manage your variety of python environments. Pyenv can be installed by executing:
```
$ curl https://pyenv.run | bash
```
Additional help for setting up pyenv can be found [here](https://realpython.com/intro-to-pyenv/)

Once you have pyenv installed add in your preferred version of Python 3.6, or simply default to Python 3.6.9 (the latest version before the release of 3.7) and set pyenv to use this version for this project. 

With the project cloned to you local system, change directory into the project's top level. In order to install all of the project dependencies to the virtual environment execute:
```
$ poetry install
```

And in order to run this project simply execute:
```
$ poetry run python csci508_final/main.py
```

## Using git
git flow is going to be critical for development on this project.

### Fancy git repo dispaly for bashrc
In order to display the current local branch to you bash terminal display execute the following to open your `bashrc` settings: 
```
$ gedit ~/.bashrc
```

Add the following snippet to the bottom of your bash rc file, be careful not to delete anything:
```
# Fancy git terminal parsing 
parse_git_branch() {
     git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}
export PS1="\u@\h \[\033[32m\]\w\[\033[33m\]\$(parse_git_branch)\[\033[00m\] $ "
```

### Branches and branching best practices
For the sake of preserving correct operation and successes `master` will be reserved for functioning and successful code that we want to be saved going forward. 
Ongoing development will be pushed to `develop` and this branch will be where testing is done before we merge `develop` into `master`
When developing new code it is best practice to branch off of `develop` and give your branch a descriptive name to describe the scope of the particular feature or improvement. There are many ways to go about creating branches, but one of the easiest ways is to use github itself. Tyler is happy to provide some additional information on this. 

### Your development path
When developing a new feature or continuing devleopment using version control and this collaborative style of development the biggest thing to remember is that we need to all be using each other's changes (unless there is a good reason not to). This cloud based repo provides a central ground truth upon which new development will be based. When you change code on your machine, even if it is in a branch, it will not automatically update the `remote` version on this repository. This `remote` versus `host` separation is the key concept when developing collaboratively. 

#### Updating your local code
In order to update your local version of the git history tree use the following `bash` command:
```
$ git fetch
```

In order to change branches to a different branch of name `my-branch`:
```
$ git checkout my-branch
```
Bear in mind that this only works when your local system is aware of the changes so be sure to `fetch` before trying to `checkout` a different branch.

Now that your local machine is synced with the `remote` host your machine is now aware of all of the new and close branches. Updating your local code to reflect what is hosted on the `remote` repository:
```
$ git pull
```

In order to merge a different branch into your local branch the following pattern may be helpful. An example repository named `my-branch` hosted locally can be merged in using:
```
$ git merge my-branch
```

If the branch is hosted on the `remote` repository:
```
$ git merge origin/my-branch
```

Once you have the local code in a state where you have accounted for changes made by everyone else you are now ready to start developing! 

#### Updating the remote code 
When you are ready to save your changes to the `remote` host you can see the files that you have added and modified by using:
```
$ git status
```

In order to stage your changes you must first add them:
```
$ git add -p
```
While there are many flags you can pass to `git add` to produce different behavior, the `-p` flag allows you to review your changes in small segments where each one must be re-approved one at a time instead of blindly adding large chunks of potentially messy or incorrect code. 

Once the code is staged you must add a tag to the the updates for bookkeeping and this is done by:
```
$ git commit -m "Some descriptive message telling everyone else what you did"
```

Now the code is finally ready to send up to the cloud. This is done by: 
```
$ git push
```

At this point, the code should be reflected on the remote repository and can be `pulled` and worked on by your team. 

#### Alterantively
If this is all a little more than you are willing to tackle there are some GUI clients that can help you visualize and manage this flow. I highly reccommend `gitkraken`. 
https://www.gitkraken.com/

This tool is great for getting started understanding git flow. If you have any questions or problems please do not hesitate to ask!

