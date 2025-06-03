# news-classifier

This repo shows how a [DistilBERT model](https://huggingface.co/distilbert/distilbert-base-uncased) 
can be used on the [AG News Classification dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset).
This dataset holds over 1 million news articles, classified as world, sports, business or science.

This repo also shows how the trained model can be exported as onnx file and quantized to make it smaller and faster.

Any questions, comments, or remarks about this project can be submitted here:
[github issues page](https://github.com/PeterPyPan/news-classifier/issues)

## Table of Contents
- [Pre-requisites](#pre-requisites)
- [Installation](#installation)
- [Usage](#usage)


## Pre-requisites
- `python` must be installed
- `make` must be installed

## Installation
Use the commands below to do the initial development setup of the repo:
```
# Clone the repo from the repo url.
git clone https://github.com/PeterPyPan/news-classifier

# Change directory into the repo.
cd news-classifier

# run make install to create a virtual environment, 
# install requirements and install pre-commit hooks
make install
```

## Usage

### Train a model
Call the model training script and provide the optional hyperparameters.
The model will be saved in `./data/output/<timestamp>`.
```
python ./src/train.py --batch_size 32 --learning_rate 2e-5 --weigth_decay 0.01 --epochs 3
```

### Evaluate the trained model
Run the evaluate script to see the perfomance of the trained model on the test set for all classes.
The evaluation metrics will be printed.
```
python ./src/evaluate.py --model_path ./data/output/<timestamp>
```

### Export trained model to onnx
Run the export onnx script to export the model as onnx and to quantize it.
The onnx files and necessary tokenizer data will be saved in `./models`.
```
python ./src/export_onnx.py --model_path ./data/output/<timestamp>
```

### Predict on example data
Run the predict script to see the predictions for some example data.
```
python ./src/predict.py
```
