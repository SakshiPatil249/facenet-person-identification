# Person-identification using Facenet
This is a repository for Facenet model in Keras, pretrained on imagnet & MS-Celeb-1M dataset 

 Model weights were initialized using parameters ported from David Sandberg's tensorflow facenet repo.

## Quick start

1. Install
* #With pip:  pip install keras-facenet
 
2. Run on Terminal: 
* On terminal give the 2 input images path & then the output will be displayed in form of distances.
* If the distance is smaller , images are same.
* If the distance is larger , images are different.

## Approach 
* We have used Keras facenet model to extract embeddings from the input image.
* Compared the embeddings of two images.
* Calculated the difference between them.
* Then we fixed the threshold value of difference.
* If the difference calculated is less than or equal to threshold then we consider it as same images.
* If the difference calculated is greater than threshold then we consider it as different images.
