# Classranker

This repository holds the code for training and preparing the model for <em>RELION class ranker</em>.


## Make dataset
To create a dataset, you need a STAR-file containing the following columns:

- ClassScore - Ground truth score
- SubImageStack - Input image path and index
- NormalizedFeatureVector - Input feature vector

See EMPIAR data base (entry-ID 10812) for more details on how to generate datasets from raw data.

To train the model the STAR-file and all the images need to be preprocessed and serialized. Use make_dataset.py to do this:

`python make_dataset.py <dataset root> <star file> --nr_valid <nr valid> --output <output file>`

In the above, `<dataset root>` is the root directory of the references in the STAR-file (`<star file>`). `<nr valid>` is the number of images that will be reserved for validation.

## Train model

After generating the dataset file above. Training can be done using the train.py, as follows:

`python train.py <dataset file> --output <log directiry> --gpu 0`

The train.py file is self-contained, so it can be copied to the log-directory of each training session and stored there. The output file is a 
