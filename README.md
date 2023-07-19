# Classranker

This repository holds the code for training and preparing the model for <em>RELION class ranker</em> as well as the executable python dependency used by <em>RELION<em>.

## Quick summary
The class ranker is part of the cryogenic electron microscopy (cryo-EM) dataset processing pipeline. It is used to automatically select suitable particles (EM images) assigned to 2D class averages for further downstream processing.

This model comprises two main components: a CNN responsible for extracting image features, and an MLP that combines the image features with additional statistics to assigns a score ranging from zero to one for each 2D class average.

The selection is subsequently done in RELION through a user defined cutoff for the predicted score.

For training, a supervised approach is adopted, where pairs of 2D classes and corresponding human-assigned scores are used to teach the model.

For more details see [this paper](https://portlandpress.com/biochemj/article/478/24/4169/230248/New-tools-for-automated-cryo-EM-single-particle).

## Make dataset
To create a dataset, you need a STAR-file containing the following columns:

- ClassScore - Ground truth score
- SubImageStack - Input image path and index
- NormalizedFeatureVector - Input feature vector

See EMPIAR data base (entry-ID 10812) for more details on how to generate datasets from raw data.

To train the model the STAR-file and all the images need to be preprocessed and serialized. Use make_dataset.py to do this:

`python train/make_dataset.py <dataset root> <star file> --nr_valid <nr valid> --output <output file>`

In the above, `<dataset root>` is the root directory of the references in the STAR-file (`<star file>`). `<nr valid>` is the number of images that will be reserved for validation.

## Train model

After generating the dataset file above. Training can be done using the train.py, as follows:

`python train/train.py <dataset file> --output <log directiry> --gpu 0`


## Citation
Published in Biochemical Journal 2021 (Volume 478, Issue 24). Bibtex:
```
@article{kimanius2021new,
  title={New tools for automated cryo-EM single-particle analysis in RELION-4.0},
  author={Kimanius, Dari and Dong, Liyi and Sharov, Grigory and Nakane, Takanori and Scheres, Sjors HW},
  journal={Biochemical Journal},
  volume={478},
  number={24},
  pages={4169--4185},
  year={2021},
  publisher={Portland Press Ltd.}
}
``` 