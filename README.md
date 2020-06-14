This is an attempt to detect covid-19 from Chest Radiography Images by using a Custom CNN, VGG16, ResNet101 and Xception. The work is heavily influenced by this tutorial at https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/ and utilizes this https://www.kaggle.com/tawsifurrahman/covid19-radiography-database dataset hosted at Kaggle.

Some details about the project,

1. "detection-of-covid-19-using-chest-x-ray.ipynb" is the main iPython file where all the dataset preparation, model generating and fitting and performance check happens.

2. Results of training such as ACC, Precision, Recall can be found at "results.txt"

3. Confusion Matrix and History graphy of training and validation can be found at "history_confusion_graph" folder

 4. "predict" folder contains some test images, a prediction program to predict test images and visualize the activation process using gradCAM. It also has a initial test result using the trained VGG16 net in the "vgg_test.png" image.
 
Trained model couldn't be uploaded due to githubs file uploading size limitation. The model can be forked and trained models can be downloaded at this link at Kaggle.
https://www.kaggle.com/amitbiswas/detection-of-covid-19-using-chest-x-ray
