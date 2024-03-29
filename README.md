
_**Text Spotter: Robust Single-Line Text Detection**_

Welcome to the CRNN implementation for text detection on the TRSynth100K dataset, sourced from Kaggle. This repository provides  tools to train a Convolutional Recurrent Neural Network (CRNN) to identify text within images.

**Installation**
Get started by installing the necessary dependencies. Simply execute the following command:
pip install -r requirements.txt

**TRSynth100K Dataset**
The TRSynth100K dataset comprises 100,000 images containing various textual content. Each image is sized at 40x160 pixels. The text within these images serves as the ground truth labels. Your objective is to accurately detect and identify the text within these images.
Download the dataset from Kaggle, unzip the file, and place the resulting folder named TRSynth100K inside the data directory.

**Data Preprocessing**
Prepare the dataset for training by preprocessing the images. Execute the following command:
python preprocessing.py

**Training the Model**
Train the CRNN model on the preprocessed dataset using the following command:
python train.py

**Making Predictions**
Once the model is trained, we can utilize it to make predictions on new images. Simply run:
python predict.py --test_img path_to_test_img
Replace path_to_test_img with the path to the image you want to analyze.

Feel free to explore, experiment, and enhance the capabilities of your CRNN model for text detection using this repository. If you encounter any issues or have suggestions for improvement, please don't hesitate to reach out. 

_**Happy coding!**_
