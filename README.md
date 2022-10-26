# Deep convolutional network for saliency prediction, implemented by PyTorch
This is a pytorch implementation of [saliency-2016-cvpr ](https://arxiv.org/abs/1603.00845) 



## Paper Abstract
The prediction of salient areas in images has been traditionally addressed with hand-crafted features based on neuroscience principles. This paper, however, addresses the problem with a completely data-driven approach by training a convolutional neural network (convnet). The learning process is formulated as a minimization of a loss function that measures the Euclidean distance of the predicted saliency map with the provided ground truth. The recent publication of large datasets of saliency prediction has provided enough data to train end-to-end architectures that are both fast and accurate. Two designs are proposed: a shallow convnet trained from scratch, and a another deeper solution whose first three layers are adapted from another network trained for classification. To the authors knowledge, these are the first end-to-end CNNs trained and tested for the purpose of saliency prediction



## Model
Two convnets are proposed in the paper, a shallow network and a deep one. here, I implemented the deep convnet with pytorch and reduced the channels of convolution layers by half to train the model faster. The main network architecture can be seen in the figure below.

 <img src="https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/figs/deep.png" width="500" height="400" class="centerImage">
 

## Results
![](https://github.com/hoseinAzdmlki/saliency-pytorch/blob/master/results/im1.png)
Note that other results may not be as good as these. But probably you be able to generate more precise maps by increasing the number of channels to the main values. Spending more time tuning the model hyperparameters can also be effective.


## Dataset
[CAT2000](http://saliency.mit.edu/results_cat2000.html) dataset is used to train the model. You can use this link to download dataset
```
http://saliency.mit.edu/trainSet.zip
```


## References 
This code draw lessons from:<br>
https://github.com/Goutam-Kelam/Visual-Saliency/tree/master/Deep_Net<br>
https://github.com/imatge-upc/saliency-2016-cvpr
