# TML_1

Aim
The aim is to predict whether each sample in the private dataset was used during model training.
The output is a CSV file containing an id and a membership score

Model Used : ResNet-18

Changes made
I changed the random submission part into a proper membership inference attack and used the trained model to predict each image from the private dataset also calculated the cross-entropy loss for every private sample.
I saved all image IDs and their membership scores into submission.csv.
