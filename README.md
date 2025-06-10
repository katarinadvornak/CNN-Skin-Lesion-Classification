How to use the custom CNN:

First download the folders with images from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T 

For the testing purposes we recommend using images size 28x28 (it works quickly) and running the version with class weights since it is the simplest and does not require additional 
augmenting images, uisng the features or loading the preprocessed model VGG-16.

--Version with class weights (training)--
1) Go to class training_with_weights.py
2) Fill the paths for your images and select all the parameters like: number of epochs, batch size, early stopping, learning rate adjuster 
3) In class loadimages.py select the size of the images you want to load
4) Run the training class and wait for the results on validation set

--Version with augmentation (training)--
1) Go to the class AugmentedImageGenerator
2) Fill the necessary paths, destination folder for the output and parameters like image size and the transformations for augmented images like rotation_range=10 etc.
3) Run this class to augment the images
4) Go to class training_with_augmentation.py
5) Fill the paths for your images and select all the parameters like: number of epochs, , batch size, early stopping, learning rate adjuster 
6) In class loadimages.py select the size of the images you want to load (the same size as augmented images have) !
7) Run the training class and wait for the results on validation set

--Testing on unseen data-- 
1) Go to class cnn_tester.py
2) Fill the necessary paths for testing images at the bottom and image sizes
3) Run the class and wait for results

--Predicting one specific image--
1) Go to class Main.py
2) Fill the necessary paths for images, used model and ID of specific image.
3) Change the image size in predictinglogic.py to desired 
4) Run the class Main.py and wait for prediction