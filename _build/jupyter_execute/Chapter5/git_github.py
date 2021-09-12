#!/usr/bin/env python
# coding: utf-8

# ## Git and GitHub

# This section shows some tricks and tools to work with Git and GitHub

# ### GitHub CLI: Brings GitHub to Your Terminal	

# If you create a local folder before creating a GitHub repository for it, you might need to go to the GitHub website then create a new repository, then add a remote to your current folder. Is there any way that all of these steps could be done in the terminal in 1 line of code?
# 
# That is when GitHub CLI  comes in handy. The code snippet below shows how to create a new Github repository in your local folder.

# ```bash
# $ cd your_local_folder
# 
# # Create an empty local git repo
# $ git init
# 
# # Create a new GitHub repo
# $ gh repo create
# ```

# With GitHub CLI, you can also manage your pull requests, issues, repositories, gists, and so much more! Check out GitHub CLI [here](https://cli.github.com/).

# ### Pull One File from Another Branch Using Git
# 

# Pull the files from another branch into your branch can be messy. What if you just want to pull one file from another branch? You can easily to like that with the code snippet below:
# 

# ```bash
# # downloads contents from remote repository
# $ git fetch
# 
# # navigate to another branch
# $ git checkout
# 
# # adds a change in the working directory to the staging area
# $ git add 
# 
# # captures the state of a project at that point in time
# $ git commit
# ```

# Now you just update one file in your branch without messing with the rest!
# 

# ### Download a File on GitHub Using wget
# 

# If you want to download a file on Github such as a csv file, instead of cloning the repo, simply use the code snippet above. For example, if the website the file you want to extract is https://github.com/khuyentran1401/Data-science/blob/master/visualization/dropdown/population.csv, type:
# 
# ```bash
# $ wget https://raw.githubusercontent.com/khuyentran1401/Data-science/master/visualization/dropdown/population.csv
# ```
# 
# Now the data is in your directory!
# 
# 

# ### github1s: Read GitHub Code with VS Code on your Browser in One Second
# 

# Syntax highlighting in VS Code makes it easy for you to understand the source code. Is there a way that you can read GitHub code with VS Code in 1s?
# 
# Yes, there is. Simply replace `github.com/yourname/repository` with `github1s.com/yourname/repository` to view Github with VS Code. It should show up like below.
# 
# [Here](https://github1s.com/khuyentran1401/Data-science) is an example.

# ![image](github1s.png)

# ### Astral: Organize Your Github Stars with Ease

# I have been sharing many tools and you might not be able to keep up with them. To save and organize the tools you discovered on Github, you can use Astral.
# 
# With Astral, you can organize my starred Github repositories like below.
# 
# I have been using this for months, and it is super quick when I want to find the right tool for my project.

# ![image](astral.png)

# [Link to Astral](http://astralapp.com/).

# ### pip install -e: Install Forked GitHub Repository Using Pip

# Sometimes, you might fork and make some changes to the original GitHub repository. If you want others to install your forked repo, use `pip install -e git+https://github.com/username/package.git#egg=package`.
# 
# The code below shows how to install the forked repo of NumPy.
# 
# ```bash
# $ pip install -e git+https://github.com/khuyentran1401/numpy.git#egg=numpy 
# ```
