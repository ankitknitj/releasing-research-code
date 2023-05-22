
# Meta Compression: Learning to compress Deep Neural Networks


This repository is the official implementation of Meta Compression: Learning to compress Deep Neural Networks
<img width="1181" alt="Screenshot 2023-05-19 at 16 28 18" src="https://github.com/ankitknitj/readme/assets/36541665/4491d1a7-e976-4977-8822-f886acb5a2df">

**Abstract.** _Deploying large pretrained deep learning models is hindered by the limitations of realistic scenarios such as resource constraints on the user/edge devices. Issues such as selecting the right pretrained model, compression method, and compression level to suit a target application and hardware become especially important. We address these challenges using a novel meta learning framework that can provide high quality recommendations tailored to the specified resource, performance, and efficiency constraints._

_For scenarios with limited to no access to unseen samples that resemble the distribution used for pretraining, we invoke diffusion models to improve generalization to test data and thereby demonstrate the promise of augmenting meta-learners with generative models. When learning across several state-of-the-art compression algorithms and DNN architectures trained on the CIFAR10 dataset, our top recommendation shows only 1% drop in average accuracy loss compared to the optimal compression method. This is in contrast to 25% average accuracy drop achieved by selecting the single best compression method across all constraints._


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
Install Hessiann-eigenthings using:

```setup
pip install --upgrade git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git@master#egg=hessian-eigenthings
```
Download the CIFAR-10 Dataset at : https://www.cs.toronto.edu/~kriz/cifar.html
Download the imagenet dataset at :
Download the augmented CIFAR-10 dataset at :

For Generating Augmented CIFAR-10 dataset, we have used Diffusion based generative models (https://github.com/NVlabs/edm).
To generate the data, Run:

```setup
git clone https://github.com/NVlabs/edm.git
cd edm
python generate.py --class=0 --outdir=out --seeds=0-999 --batch=1000 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
```
where **class** refers to the CIFAR-10 classes, and **seeds** and **batch** can be altered to generate desired number of images. We have generated 1000 images of each classes (10K images in total). 



## Metadata Extraction

For CIFAR-10 based experiments, first the pretrained models (originally trained on Imagenet dataset) are fine tuned on CIFAR-10 dataset.

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

These models are available at : 

For Metadata extraction using these pre-trained classifiers, run:

**Note**. _Before running the below script, change the path in **main_prepare_metadata** function to point to the path where you want to store the extracted metadata._ _Also, in the **prepare_cifar_modeldict** function, the checkpoints () , needs to be loaded as well._ _Similarly, the dataset path () in the functions **prepare_imagenet_dataset** and **prepare_cifar_dataset** , needs to be changed to the path where you have downloaded the datasets._

```train
python extract_metadata.py
```
This will generate Train and Test metadata which contains imformation about performance of compressed classifier on CIFAR-10 Eval (Data from diffusion models) and Original CIFAR-10 Test dataset. 

Metadata refers to the dataset containing certain meta features including architechture features, compression method features, norms of weights, gradients, loss and accuracy of compressed classifier, etc. 

**Note.** For Imagenet experiments, just uncomment the **prepare_imagenet_dataset()**, **prepare_imagenet_modeldict()** and **dummy input** definition in _extract_metadata.py_


## Training of Accuracy prediction model and Evaluation

We use gradient boosted decision trees as the accuracy prediction model due to their good performance in regression tasks for small datasets.

We report the generalisation performance of our trained meta predcitor to new architechtures as well as new compression methods. For generalisation to new architechtures, make the flag **arch_split** to **True**, and for generalisation to new compression methods, make the flag **compr_split** to **True**. 

To train and evaluate the XGBoost model on the extracted metadata, run:

**Note.** _Before running the below script, change the path in prepare_dataset function to the location where your extracted metadata is stored. _
```eval
python meta_prediction.py
```


## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
