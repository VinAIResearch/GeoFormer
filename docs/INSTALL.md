# Installation guide

### Requirements
* Python 3.7.0
* Pytorch 1.7.1
* CUDA 10.2

1\) Clone this repository

```
git clone https://github.com/VinAIResearch/GeoFormer
cd GeoFormer
```

2\) Install pytorch (version 1.7.1), cudatoolkit (version 10.2) and other dependencies
```
conda install pytorch==1.7.1 cudatoolkit=10.2 -c pytorch

conda install -c bioconda google-sparsehash 

pip install -r requirements.txt

```

We do not recommend to use newer version of Pytorch due to the lack of THC library. 

3\) For the SparseConv, we use spconv1.0 from [PointGroup](https://github.com/llijiang/spconv/tree/740a5b717fc576b222abc169ae6047ff1e95363f)

* To compile `spconv`, firstly install the dependent libraries. 
```
conda install libboost
conda install -c daleydeng gcc-5 # need gcc-5.4 for sparseconv
```
Add the `$INCLUDE_PATH$` that contains `boost` in `lib/spconv/CMakeLists.txt`. (Not necessary if it could be found.)
```
include_directories($INCLUDE_PATH$)
```

* Clone and compile the `spconv` library.
```
cd lib/
git clone https://github.com/llijiang/spconv.git --recursive
cd spconv/
python setup.py bdist_wheel
```

* Run `cd dist` and `pip install` the generated `.whl` file.

Currently, there are some bugs with spconv2.0. We are planning to refactor and optimize our model to run with spconv2.0.

4\) Compile the `pointgroup_ops` library.
```
cd lib/pointgroup_ops
python setup.py develop
```
If any header files could not be found, run the following commands. 
```
python setup.py build_ext --include-dirs=$INCLUDE_PATH$
python setup.py develop
```
`$INCLUDE_PATH$` is the path to the folder containing the header files that could not be found.

5\) Compile the `pointnet2` library.
```
cd lib/pointnet2
python setup.py install
```

6\) Install FAISS:

```
conda install -c faiss-gpu cudatoolkit=10.2 # for CUDA 10.2
```
