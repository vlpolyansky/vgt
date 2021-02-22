# Extra scripts

### Additional requirements

Python scripts require:
- numpy
- [gudhi](https://gudhi.inria.fr/python/latest/installation.html)
- [miniball](https://pypi.org/project/miniball/)
- [tqdm](https://github.com/tqdm/tqdm)

C++ scripts have their optional targets in `make` and can be compiled during the main cmake-make routine.

### Available scripts
|Script|Description|
|---|---|
|**Natural image patches**| Scripts to process the *van Hateren's Natural Image Dataset*|
|`imc2npy.py`| A script to combine all van Hateren images into a single (large, about 12Gb) numpy file.<br>The data and essentially the code are from http://bethgelab.org/datasets/vanhateren/.|
|`generate_patches.py`| A script to cut the images into 3x3 patches. All the arguments are hard-coded in the beginning of the file, e.g. how many patches to extract and how many to filter out based on their D-norm. In the end, a gaussian noise is added to the patches. This process is described in [The Nonlinear Statistics of High Contrast Patches in Natural Images](https://dash.harvard.edu/bitstream/handle/1/3637108/mumford_nonlinstatpatches.pdf?sequence=1).|
|`prepare_patches.cpp`| Command example:<br> `./vgt/prepare_patches patches.npy -n 5000 -k 4 -t 30`, which would perform filtration by density that corresponds to the definition of `M0[k, T]` in [Topology and Data](https://www.ams.org/journals/bull/2009-46-02/S0273-0979-09-01249-X/S0273-0979-09-01249-X.pdf) and `n` can be used to subsample the initial set of patches.|
|**TDA**||
|`compute_persistence.py`| Takes data and the output of `approx_delaunay` to compute persistence diagrams of the corresponding approximated Delaunay-Čech complex, saving the persistence pairs in `--out`. <br>If computing filtration takes too much time, we suggest creating a C++ implementation using CGAL to compute Čech filtrations. Gudhi::Simplex_tree might be also helpful.|