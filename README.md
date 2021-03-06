# PipelineX PyTorch

An example project using [PipelineX](https://github.com/Minyus/pipelinex), Kedro, PyTorch, Ignite, and SHAP for image classification

<p align="center">
<img src="img/kedro_pipeline.PNG">
Pipeline visualized by Kedro-viz
</p>

## 1. Install dependencies

```bash
$ pip install pipelinex torch torchvision pytorch-ignite kedro shap mlflow kedro-viz
```

Note: `mlflow` and `kedro-viz` are optional.

## 2. Clone this repository and run `main.py`

```bash
$ git clone https://github.com/Minyus/pipelinex_pytorch.git
$ cd pipelinex_pytorch
$ python main.py
```

## Tested environment

- Python 3.6.8


## Simplified Kedro project template

This project was created from the GitHub template repository at https://github.com/Minyus/pipelinex_template

To use for a new project, fork the template repository and hit `Use this template` button next to `Clone or download`.

<p align="center">
<img src="https://help.github.com/assets/images/help/repository/use-this-template-button.png">
</p>
