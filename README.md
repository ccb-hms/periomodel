<p float="left">
    <img style="vertical-align: top" src="./images/HMS_DBMI_Logo.png" width="50%" />
</p>

## Estimating Periodontal Stability Using Computer Vision ## 

This repository contains the complete code base that was used to 
train and evaluate the periodontal stability model as described in [1].
The following notebooks are provided to demonstrate how the model is used
to predict the disease classes on the test cases.

- [Download model weights and test images](./notebooks/01-download-data.ipynb)
- [Run the model on the test data](./notebooks/02-run-model.ipynb)

The execution of the notebooks require model weights and test data which can
be downloaded from an S3 bucket. Contact 
the authors of the manuscript to request access to the data.

## Installation with Docker ##
The most convenient way to get started with this repository is to run the 
notebooks in a [Docker](https://docs.docker.com/) container. 
The included [Dockerfile](Dockerfile) builds a container image with a reproducible Python 
development environment. The docker image is based on 
Debian GNU Linux distribution with Python 3.11.9. All libraries that were 
used to train and evaluate the model are included in the docker image.

Here's a step-by-step guide on how to use this setup:

1. Install [Docker](https://docs.docker.com/) on your machine.
2. Clone the GitHub project repository to download the contents of the repository:
```bash
# Clone the repository in the local environment
git clone git@github.com:ccb-hms/periomodel.git
```
3. Navigate to the repository's directory to change your current directory to the repository's 
directory.
4. Build the Docker image. Use the command `docker compose build` to build a Docker image from the 
Dockerfile in the current directory. 
```bash
# Change into the repository's directory
cd periomodel
# Build the docker image
docker compose build
```
5. Edit the last line in the `docker-compose.yml` file to map a local data directory (for example: 
`/user/username/data` to the container image:
the container:
```aiignore
# Edit the last section of the docker-compose.yml file
volumes:
      - .:/app
      - /user/username/data:/app/data
```
6. Run `docker compose up` to start the Docker container based on the configurations 
in the docker-compose.yml.
```bash
docker compose up
```
This command also starts a jupyter lab server inside the container which can be accessed
from a browser by clicking on the link displayed.

### GPU support for Docker ###

The container does not require a GPU to run, but the `docker-compose.yml` file can be
modified to allow access to the GPU driver from inside the container.
For detailed instructions on how to set this up, refer to the 
[Docker documentation](https://docs.docker.com/compose/gpu-support/). The NVIDIA Container Toolkit is a set of tools designed to enable GPU-accelerated applications to run 
within Docker containers. It facilitates the integration of NVIDIA GPUs with container runtimes, 
allowing developers and data scientists to harness the power of GPU computing in containerized environments.
See the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) page for installation instructions.

## Installation without Docker ##

For installation in a local environment we provide [Pipfile](Pipfile) and [Pipfile.lock](Pipfile.lock) files 
which are used to produce deterministic builds. These file are needed by
[Pipenv](https://pipenv.pypa.io/en/latest/index.html), a Python virtualenv management tool that
integrates pipenv and virtualenv.
```bash
# Create a pipenv environment with all dependencies
pipenv install -e . --dev
# Run jupyter lab using the docker entry script
pipenv run ./bash_scripts/docker_entry
```
The notebooks use an environment variable called `DATA_ROOT` to keep track of the 
location of the data files. For use with a docker container, the variable is defined in the Dockerfile
as `DATA_ROOT=/app/data` (see above). When using Pipenv as package manager, it is 
defined in the [.env](.env) file which is automatically loaded into the environment after activation.

## Annotation of dental radiographs ##

[Label Studio](https://labelstud.io/) is an open-source data labeling tool for labeling, annotating, 
and exploring many different data types. Additionally, the tool includes 
a powerful machine learning interface that can be used for new model training, 
active learning, supervised learning, and many other training techniques.

1. Multi-type annotations: Label Studio supports multiple types of annotations, including labeling for audio, video, images, text, and time series data. These annotations can be used for tasks such as object detection, semantic segmentation, and text classification among others.
2. Customizable: The label interface can be customized using a configuration API.

<img src="./images/LabelInterface.png" width="70%" height="70%"/>

3. Machine Learning backend: Label Studio allows integration with machine learning models. You can pre-label data using model predictions and then manually adjust the results.
4. Data Import and Export: Label Studio supports various data sources for import and export. You can import data from Amazon S3, Google Cloud Storage, or a local file system, and export it in popular formats like COCO, Pascal VOC, or YOLO.
5. Collaboration: It supports multiple users, making it suitable for collaborative projects.
6. Scalability: Label Studio can be deployed in any environment, be it on a local machine or in a distributed setting, making it a scalable solution.

### How to Use Label Studio
The tool is included in this repository as a [submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules).
Please see the [CCB Computervision Repository](https://github.com/ccb-hms/computervision) 
installation instructions.