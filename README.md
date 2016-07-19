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
> python copy_conv_rec_learner.py <training_file> <max_num_epochs> <D> <test_file>
```
were `D` is the embedding space dimenssion (128 in paper.)
The best model will be saved at `<training_file>.pkl`

To evaluate an existing model re-run with exactly the same parameteres except
for `<max_num_epochs>` which should be zero.

The following code will generate names from a pre-trained model and a test_file
with code examples.

```python
model = ConvolutionalCopyAttentionalRecurrentLearner.load(model_fname)
test_data, original_names = model.naming_data.get_data_in_recurrent_copy_convolution_format(test_file, model.padding_size)
test_name_targets, test_code_sentences, test_code, test_target_is_unk, test_copy_vectors = test_data

idx = 2  # pick an example from test_file
res = model.predict_name(np.atleast_2d(test_code[idx]))
print "original name:", ' '.join(original_names[idx].split(','))
print "code:", ' '.join(test_code[idx])
print "generated names:"
for r,v in res:
    print v, ' '.join(r)
```