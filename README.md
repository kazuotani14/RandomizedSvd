Implementation of Fast Randomized SVD [1] for low-rank approximation of matrices. C++/Eigen, interface is same as Eigen's jacobiSVD.

Also a demo of background removal in webcam feed using randomized SVD.

### Requirements

* Eigen (currently assumed to be in _eigen_ folder in root)
* OpenCV (for background removal demo)

### References

1. Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, _Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions_, 2009 ([available on arXiv](http://arxiv.org/abs/0909.4061>`))
2. [Facebook implementation of Fast Randomized SVD](https://github.com/facebook/fbpca) (python/numpy)
3. fast.ai [Numerical Linear Algebra course](https://github.com/fastai/numerical-linear-algebra)
