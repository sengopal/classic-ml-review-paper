# Classic algorithms are fair learners: Classification Analysis of natural weather and wildfire occurrences
### Introduction
Classic machine learning algorithms have been reviewed and studied mathematically on its performance and properties in detail. This paper intends to review the empirical functioning of widely used classical supervised learning algorithms such as Decision Trees, Boosting, Support Vector Machines, k-nearest Neighbors and a shallow Artificial Neural Network. The paper evaluates these algorithms on a sparse tabular data for classification task and observes the effect on specific hyperparameters on these algorithms when the data is synthetically modified for higher noise. These perturbations were introduced to observe these algorithms on their efficiency in generalizing for sparse data and their utility of different parameters to improve classification accuracy. The paper intends to show that these classic algorithms are fair learners even for such limited data due to their inherent properties even for noisy and sparse datasets.

### Citation
If you find this project useful in your research or work, please consider citing it:
```
@article{gopal2023classic,
  title={Classic algorithms are fair learners: Classification Analysis of natural weather and wildfire occurrences},
  author={Gopal, Senthilkumar},
  journal={arXiv preprint arXiv:2309.01381},
  year={2023}
}
```
### Setup Instructions
1. All the necessary files are available at <add repo link>
2. `git clone` or download the files to continue with the setup

### Datasets
a. Rattle - https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
b. Wildfire - https://www.kaggle.com/rtatman/188-million-us-wildfires

### Dataset setup
1. Rattle dataset is already available in Github. For some reason, if that needs to be updated, drop the downloaded file as `\data\rattle\weatherAUS.csv`
2. Wildfire is too large and is "not" available in Github. Download the file `FPA_FOD_20170508.sqlite` from https://www.kaggle.com/rtatman/188-million-us-wildfires and place in location - `\data\wildfire\FPA_FOD_20170508.sqlite`

### Installations
1. All the files need Python version 3.7 and above
2. The environment file has been exported as `environment.yml`
3. Use `conda env create -f environment. yml` to create the `py37` environment
4. Activate the new environment: conda activate py37
5. Incase if you believe a working py37 environment is setup and reasonably stable, check for the following libraries and their versions.
    pd.__version__ 0.25.0
    np.__version__ 1.16.4
    sklearn.__version__ 0.21.2
    seaborn.__version__ 0.9.0
    matplotlib.__version__ 3.1.0
6. An optional setup is required for Graphviz to view the decision tree images. In case the executable is not available, the code continues gracefully, with an error message.
7. Install GraphViz from https://graphviz.gitlab.io/ and ensure it is available in the PATH

### Folders and Locations
The following folders are required without which the code would fail.
1. images/rattle
2. images/wildfire
3. images/rattle/pre-process
4. images/wildfire/pre-process
5. output/results
6. output/results/rattle
7. output/results/wildfire

### Execution
1. run the python file `data_read.py` which performs all the necessary actions for all the five algorithms. The typical runtime with default variables would be around 45 minutes.

### Variables and runtime

All the following variables are available at the beginning of the `data_read.py` file
1. VALIDATION_CURVE - If set to True, the code will generate all the validation curves for different hyperparameters. This increases the execution time by atleast 1 hour.
2. GRID_SEARCH - If set to True, the code will run a grid search to find the optimal values of hyper parameter combinations instead of using the identified ones. This increases the execution time by atleast 3-4 hours. Please use this with caution.
3. EPOCH_GRAPH - If set to True, the code will generate the Learning curve using epochs for NN and SVM. This increases the execution time by atleast 1 hour.

### Note
1. Ignore any warnings generated on the console as most of them were deprecation related.
