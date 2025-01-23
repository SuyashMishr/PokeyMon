# PokeyMon
Pokémon Image Classification Using Vision Transformer (ViT) :

This repository provides a solution to classify Pokémon images using the Vision Transformer (ViT) model. The project uses deep learning techniques to classify images into different categories based on their Pokémon types.

Overview :
The goal of this project is to classify Pokémon images into different categories based on their respective types using the Vision Transformer (ViT). ViT is an advanced deep learning model originally designed for image classification tasks, which divides images into patches and processes them with self-attention mechanisms similar to those used in NLP (Natural Language Processing).

This project uses a custom dataset of Pokémon images and applies a ViT model for classification. It includes functions for data loading, preprocessing, model training, evaluation, and visualization.

Features
1.Custom Dataset: The dataset consists of Pokémon images grouped by type, which are used to train and evaluate the ViT model.
2.Model: The project uses the pre-trained google/vit-base-patch16-224-in21k model from Hugging Face, which is fine-tuned for this image classification task.
3.Data Augmentation: Images are preprocessed with resizing, normalization, and tensor conversion to feed into the model.
4.Training: The model is trained using the AdamW optimizer with a cross-entropy loss function.
5.Evaluation: The model is evaluated based on its test accuracy and loss, and the results are displayed for analysis.
6.Visualization: Training and validation losses are plotted to help analyze the model's performance over time.
Prerequisites
#Before running the code, make sure you have the following libraries installed:

1.torch
2.transformers
3torchvision
4.datasets
5.matplotlib
6.numpy
7.tqdm

You can install them via pip if you haven't already:

1.bash
2.Copy
3.Edit
4.pip install torch transformers torchvision datasets matplotlib numpy tqdm
5.Dataset

The dataset used in this project consists of images of Pokémon grouped into folders based on their type. The dataset is split into training, validation, and test sets for model training and evaluation.

Ensure that the dataset directory structure is as follows:

bash:

/dataset
    /train
        /class_1
        /class_2
        ...
    /val
        /class_1
        /class_2
        ...
    /test
        /class_1
        /class_2
        ...
        
Each subdirectory under train, val, and test represents a different Pokémon type.

How to Run :
Step 1: Prepare the Dataset
First, ensure that the images are properly organized into train, val, and test directories. If the dataset needs to be split or moved, the code automatically handles this by organizing the images into respective directories.

Step 2: Configure the Model and Preprocessing
The model is configured to use the Vision Transformer (ViT) architecture. Preprocessing functions, including resizing, normalization, and tensor conversion, are applied to prepare the images for training.

Step 3: Train the Model
Once the data is ready, simply run the training loop. The model will be trained for a specified number of epochs (5 epochs by default) and the training/validation loss will be displayed at each epoch.

python code

train_loss, val_loss = train_model(model, train_loader, val_loader, epochs)
Step 4: Evaluate the Model
After training, the model is evaluated on the test set to calculate the accuracy and loss.

python code

test_loss, test_accuracy = evaluate_model(model, test_loader)
Step 5: Visualize the Results
The training and validation losses are plotted over the epochs to help you understand the model's learning progress.

python code

plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()
Results

After training and evaluation, the model's performance will be shown in the form of:

Test Loss: The loss on the test set.
Test Accuracy: The accuracy achieved on the test set.
These metrics will help you determine how well the model performs at classifying Pokémon types.

Model Details
Model Type: Vision Transformer (ViT)
Pre-trained Model: google/vit-base-patch16-224-in21k
Optimizer: AdamW
Loss Function: Cross-Entropy Loss
Image Size: 224x224 pixels
Batch Size: 16
Number of Epochs: 5
Customization

You can adjust the following parameters to experiment with the model:

epochs: The number of training epochs.
batch_size: The batch size used during training.
learning_rate: The learning rate for the optimizer.
img_size: The size to which the images are resized before feeding into the model.

Conclusion :
This project demonstrates the application of the Vision Transformer (ViT) model to a real-world image classification task—classifying Pokémon images by type. The code includes all necessary steps to load data, preprocess images, train the model, and evaluate its performance. Feel free to modify and experiment with the code for other image classification tasks!

