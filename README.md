# Voronoi Graph Traversal

## Introduction

The goal of this project is to alleviate the high computational cost of Voronoi diagram and Delaunay tessellation constructions in high dimensions. This can allow some new applications of geometric methods which utilize these structures on more complex data. 

The project makes use of an idea of a graph traversal, or marching, over a Voronoi graph (i.e. 1-skeleton of a Voronoi diagram) without its explicit computation, while extracting required information about the dual of the visited vertices.

We demonstrate the first applications of this methodology in our KDD'20 submission <br>[Voronoi Graph Traversal in High Dimensions with Applications to Topological Data Analysis and Piecewise Linear Interpolation](https://dl.acm.org/doi/10.1145/3394486.3403266).

## Features
*Currently planned features are also included in their preliminary order of appearance.*

- [x] General interface for interactions with the Voronoi graph (`VoronoiGraph.{cpp|h}`)
- [x] Delaunay k-skeleton extraction (`approx_delaunay.cpp`)
- [x] Piecewise-linear interpolation (`interpolate.cpp`)
- [ ] Geometric density estimation
- [ ] Power diagrams (i.e. additive weighting)
- [ ] Non-Euclidean/Riemannian metric spaces (e.g. hyperspherical)

Also, `vgt/extra` contains a few helper scripts for some experiments from the paper above.

## Technical details
### Dependencies
- C++17, cmake, make
- zlib 1.2.11 (required by cnpy)
- (_Recommended_) OpenMP (enable parallel computations)

Already included and do not require installation:
- [Eigen](https://eigen.tuxfamily.org/) for linear algebra
- [cnpy](https://github.com/rogersce/cnpy) to read numpy files in C++
- [argparse](https://github.com/p-ranav/argparse) for command line argument parsing
- [tqdm](https://github.com/tqdm/tqdm.cpp) for a progress line bar during the computations

### Compiling
A "standard cmake routine" can be applied to compile the code from sources:
```shell
mkdir build && cd build
cmake ..
make
```
You can choose to pass additional preprocessor parameters to _cmake_ to optimize your code with a string 
<br>`-DCMAKE_CXX_FLAGS="-D<key>=<value>"`. The following keys are available:
- `DIMENSIONALITY` If you have a fixed dimensionality of your data, then, following the suggestions from Eigen about [fixed size matrices](https://eigen.tuxfamily.org/dox/classEigen_1_1Matrix.html#fixedsize), you can compile the sources for that specific dimensionality to potentially speed up the computations. Default dimensionality used is `Eigen::Dynamic`.
- `FTYPE` By default, all computations are performed in `long double`. However, if you want to use some other floating-point type you can pass the corresponding flag to _cmake_, for example `-DFTYPE=double`. Be prepared that by using `double` or `float`, the numeric instability may drastically increase in high-dimensional spaces. You can control it by looking at the number of failed validations after running an algorithm.<p>
You may also opt to use multiple-precision arithmetics libraries, such as [MPIR](http://www.mpir.org/) or [GMP](https://gmplib.org/). Such libraries are not included in this package, so you would need to include them manually and update the line `using ftype = long double;` in `utils.h` with your preferred floating-point type. Using these libraries would allow one to achieve a desired numeric stability at the cost of a slower running time. 

### Executables
All executables have a `-h` command line argument to show an up-to-date information about the available command line arguments.

#### Delaunay skeleton approximation
```shell
vgt/approx_delaunay [options] data.npy
```

| Argument | Description |
| -------- | ----------- |
|`data.npy` | npy NxD data matrix of 32-bit floats |
|`--dim <int:2>` | maximum dimensionality of extracted Delaunay simplices |
|`--steps <int:1000>` | number of steps to perform in a random walk from each starting vertex; <br>a non-positive value would instead correspond to a full walk / complete graph search |
|`--noclosure` | flag to output only k-dimensional Delaunay simplices without their closure |
|`--out <str:"output.txt">` | output filename for a txt file containing Delaunay simplices; <br>each line describes a simplex in a form `d v_0 v_1 ... v_d [f]`, where <br>`v_i` are indices of data points that form the simplex and <br>`[f]` is the filtration value of the simplex, if it was computed |
|`--vout <str:"">` | optional output filename for a txt file containing visited Voronoi vertices; <br>each line describes a vertex in a form of its dual `d v_0 v_1 ... v_d` |
|`--seed <int:239>` | random seed |
|`--njobs <int:1>`| number of parallel threads (requires OpenMP)|
|`--strategy <str:"brute_force">` | ray sampling strategy, available values: `{"brute_force", "bin_search"}` |

#### Piecewise-linear interpolation
```shell
vgt/interpolate [options] data.npy query.npy
```

| Argument | Description |
| -------- | ----------- |
|`data.npy` | numpy NxD data matrix of 32-bit floats |
|`query.npy` | numpy MxD data matrix of 32-bit floats |
|`--values <str:"">` | npy Nx1 matrix of 32-bit floats |
|`--out <str:"interpolated.npz">` | output filename for an npz data file, format described below |
|`--seed <int:239>` | random seed |
|`--njobs <int:1>`| number of parallel threads (requires OpenMP)|
|`--strategy <str:"brute_force">` | ray sampling strategy, available values: `{"brute_force", "bin_search"}` |

The output npz file contains the following matrices:
- `indices ,shape=(L, )` indices of query points for which the interpolation was successful (points must lie inside the convex hull of the data)
- `simplices, shape=(L, D+1)` each line contains (D+1) indices of data points that are vertices of a Delaunay simplex containing the corresponding query point 
- `coefficients, shape=(L, D+1)` contains barycentric coordinates of query points, based on data point indices from `simplices`
- `estimates, shape=(L,)` **(if values are provided)** interpolation values of query points



## BibTeX
```
@inproceedings{polianskii2020voronoi,
  title={Voronoi Graph Traversal in High Dimensions with Applications to Topological Data Analysis and Piecewise Linear Interpolation},
  author={Polianskii, Vladislav and Pokorny, Florian T},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2154--2164},
  year={2020}
}
```

## License
The project is available under the [MIT](https://opensource.org/licenses/MIT) license.

