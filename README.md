# Computer Vision 
## Tutorial 2: Reproducible Deep Learning

This lesson introduces essential tools for developing deep learning projects to promote code *reproducibility* and *reusability*. We will cover three open-source tools: (1) [Git](https://git-scm.com/), (2) [Docker](https://www.docker.com/), and (3) [Hydra ](https://github.com/facebookresearch/hydra). The choice of tools is questionable but offers a good compromise between practicality and educational needs.

## Local set-up
You can run this project in any local environment. Just as an example, you may choose an [Anaconda](https://www.anaconda.com/products/individual) virtual enviroment. If so, you can create a virtual environment with the following command:
```bash
conda create -n reprodl python=3.12 pip; conda activate reprodl
```
and install all the requirements.
```bash
pip install -r requirements.txt
```

## Steps
You can move through existing branches using Git.
```bash
git checkout existing_branch
```

We will end up with the following branches.
| Branch | Content |
| ------------- |------------- |
| main | It's our starting point, which contains just a notebook. |
| step1_going_modular | In this step, we will rewrite our notebook to make our code modular. We will end up moving all code in separate files. |
| step2_docker | We introduce a Dockerfile and isolate our code from the OS. |
| step3_hydra | In this final step, we want to make our code more easily configurable. We will remove all variables and move them to a dedicated file. |

### Optional steps
In this lesson, we saw some very simple tools with which you can improve the reproducibility, portability, and reuse of your Deep Learning projects. Remember that there are dozens of other tools for every step of your project. Here are some ones:
-  [DVC](http://dvc.org/): it allows to track your data changes. You can think about it as a Git for data.
-  [Docker Hub](https://hub.docker.com/): you can publish your Docker images in public repositories the same way you publish code on Github.
-  [Weights&Biases](https://wandb.ai/): it is a very simple and powerful tool to log all your experiments and parameters on a dashboard in the cloud.
-  [MLFlow](https://mlflow.org/): it allows to keep track of your model versions as well as logging your training parameters.
