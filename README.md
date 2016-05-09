Convolutional Attention Network
===============
Code related to the paper:
```
@inproceedings{allamanis2016convolutional,
  title={A Convolutional Attention Network for Extreme Summarization of Source Code},
  author={Allamanis, Miltiadis and Peng, Hao and Sutton, Charles},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2016}
}
```
For more information and the data of the paper, see [here](http://groups.inf.ed.ac.uk/cup/codeattention/).

The project depends on Theano and uses Python 2.7.

Usage Instructions
======
To train the `copy_attention` model with the data use
```
> python copy_conv_rec_learner.py <training_file> <max_num_epochs> D <test_file>
```

