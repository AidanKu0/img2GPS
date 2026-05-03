# Image2GPS

This project predicts GPS coordinates from images using a convolutional neural network. The model takes an input image and outputs a latitude and longitude prediction. The project includes scripts for preprocessing the image data, splitting the dataset, training the model, and evaluating prediction accuracy.

## Dataset

The collected image dataset is stored separately on Hugging Face

Dataset link: [Hugging Face Dataset]([https://huggingface.co/datasets/aidankuo/img2GPStrainData](https://huggingface.co/datasets/aidankuo/img2GPSTrainingData))

After downloading the dataset, place it in the project folder with this structure:

```text
IMAGE2GPS/
  data/
    all_images/
      image1.jpg
      image2.jpg
      ...
