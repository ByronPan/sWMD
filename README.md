# sWMD
## sWMD in python


### 1.Introduction.

An efficient technique to learn a supervised metric, which we call the Supervised WMD (S-WMD) metric. To see the details, please refer http://papers.nips.cc/paper/6139-graph-clustering-block-models-and-model-free-results.pdf



### 2.Required.

* Python 2
* numpy
* scipy 0.18.1
* cython



### 3.Getting start.
Download the fold "dataset" from https://github.com/ByronPan/S-WMD.  

Make sure that all the files and documents in a same directory. First compile the code by:

```python
>>> python setup.py build_ext --inplace

```
Then run the code by:

```python
>>> ipython swmd.py

```

For a detailed list of functionality see "functions.pyx"
