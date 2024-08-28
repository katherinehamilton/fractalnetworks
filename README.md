# fractal-networks
This repository contains the code which accompanies my MSc Thesis titled *Origins of Fractality in Complex Networks*. A key component of this is the `fractalnetworks` module which consists of functions for the analysis of fractal complex networks. 

## Table of Contents
- [Getting Started](#gettingstarted)    
    - [Dependencies](#dependencies)
    - [Installation](#installation)

## Getting Started
This section explains how to install and use the `fractalnetworks` package locally, and the prerequisites you need to get started. 

### Dependencies
* matplotlib==3.8.0
* networkx==3.1
* numpy==2.1.0
* pandas==2.2.2
* python_igraph==0.11.6
* scikit_learn==1.3.2
* scipy==1.14.1
* tqdm==4.66.4

### Installation
Below is an example of how to install the module locally for personal use. Before using the module, all dependencies listed in the section above need to be installed. 
1. Open a new command prompt terminal.  
2. Create a new folder for the git repository and initialise an empty repository.
```sh
mkdir python_project_folder
cd python_project_folder
git init
```
3. Clone the repository.
```sh
git clone https://github.com/katherinehamilton/fractalnetworks.git
```
4. Import the module to python.
```sh
cd fractalnetworks
python
import fractalnetworks as fn
```
