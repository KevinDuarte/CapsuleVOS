## CapsuleVOS

This is the code for the ICCV 2019 paper CapsuleVOS: Semi-Supervised Video Object Segmentation Using Capsule Routing. 

The network is implemented using TensorFlow 1.4.1.

Python packages used: numpy, scipy, scikit-video

## Files and their use

1. caps_layers_cod.py: Contains the functions required to construct capsule layers - (primary, convolutional, and fully-connected, and conditional capsule routing).
2. caps_network_train.py: Contains the CapsuleVOS model for training.
3. caps_network_test.py: Contains the CapsuleVOS model for testing.
4. caps_main.py: Contains the main function, which is called to train the network.
5. config.py: Contains several different hyperparameters used for the network, training, or inference.
6. inference.py: Contains the inference code.
7. load_youtube_data_multi.py: Contains the training data-generator for YoutubeVOS 2018 dataset.
8. load_youtubevalid_data.py: Contains the validation data-generator for YoutubeVOS 2018 dataset.

## Data Used

We have supplied the code for training and inference of the model on the YoutubeVOS-2018 dataset. The file <code>load_youtube_data_multi.py</code> and <code>load_youtubevalid_data.py</code> creates two DataLoaders - one for training and one for validation. The <code>data_loc</code> variable at the top of each file should be set to the base directory which contains the frames and annotations.

To run this code, you need to do the following:
1. Download the YoutubeVOS dataset 
2. Perform interpolation for the training frames following the papers' instructions

## Training the Model

Once the data is set up you can train (and test) the network by calling <code>python3 caps_main.py</code>.

The <code>config.py</code> file contains several hyper-parameters which are useful for training the network. 

## Output File

During training and testing, metrics are printed to stdout as well as an output*.txt file. During training/validation, the losses and accuracies are printed out to the terminal and to an output file.

## Saved Weights

Pretrained weights for the network are available [here](https://drive.google.com/open?id=12zvvqd5i3EVNzPF-hEfq_hi2CEzRRSjS). To use them for inference, place them in the <code>network_saves_best</code> folder.

## Inference

If you just want to test the trained model with the weights above, run the inference code by calling <code>python3 inference.py</code>. This code will read in an .mp4 file and a reference segmentation mask, and output the segmented frames of the video to the Output folder.

An example video is available in the Example folder.

