
# Locally Optimized Product Quantization

## Objectives
This is a C++ port of searcher component for Locally Optimized Product Quantization (LOPQ) code, designed for CPU, originally located at https://github.com/yahoo/lopq.

A model, pretrained with code provided for Python or Spark and deployed via a Protobuf format is required to, e.g., search backends for high performance approximate nearest neighbor search.

## Overview

Locally Optimized Product Quantization (LOPQ) [1] is a hierarchical quantization algorithm that produces codes of configurable length for data points. These codes are efficient representations of the original vector and can be used in a variety of ways depending on the application, including as hashes that preserve locality, as a compressed vector from which an approximate vector in the data space can be reconstructed, and as a representation from which to compute an approximation of the Euclidean distance between points.

### Training

Conceptually, the LOPQ quantization process can be broken into 4 phases. The training process also fits these phases to the data in the same order.

1. The raw data vector is PCA'd to `D` dimensions (possibly the original dimensionality). This allows subsequent quantization to more efficiently represent the variation present in the data.
2. The PCA'd data is then product quantized [2] by two k-means quantizers. This means that each vector is split into two subvectors each of dimension `D / 2`, and each of the two subspaces is quantized independently with a vocabulary of size `V`. Since the two quantizations occur independently, the dimensions of the vectors are permuted such that the total variance in each of the two subspaces is approximately equal, which allows the two vocabularies to be equally important in terms of capturing the total variance of the data. This results in a pair of cluster ids that we refer to as "coarse codes".
3. The residuals of the data after coarse quantization are computed. The residuals are then locally projected independently for each coarse cluster. This projection is another application of PCA and dimension permutation on the residuals, and it is "local" in the sense that there is a different projection for each cluster in each of the two coarse vocabularies. These local rotations make the next and final step, another application of product quantization, very efficient in capturing the variance of the residuals.
4. The locally projected data is then product quantized a final time by `M` subquantizers, resulting in `M` "fine codes". Usually the vocabulary for each of these subquantizers will be a power of 2 for effective storage in a search index. With vocabularies of size 256, the fine codes for each indexed vector will require `M` bytes to store in the index.

The final LOPQ code for a vector is a `(coarse codes, fine codes)` pair, e.g. `((3, 2), (14, 164, 83, 49, 185, 29, 196, 250))`.

*Training code is not included in this repository. Please use original Python code for training process.*

### Nearest Neighbor Search

A nearest neighbor index can be built from these LOPQ codes by indexing each document into its corresponding coarse code bucket. That is, each pair of coarse codes (which we refer to as a "cell") will index a bucket of the vectors quantizing to that cell.

At query time, an incoming query vector undergoes substantially the same process. First, the query is split into coarse subvectors and the distance to each coarse centroid is computed. These distances can be used to efficiently compute a priority-ordered sequence of cells [3] such that cells later in the sequence are less likely to have near neighbors of the query than earlier cells. The items in cell buckets are retrieved in this order until some desired quota has been met.

After this retrieval phase, the fine codes are used to rank by approximate Euclidean distance. The query is projected into each local space and the distance to each indexed item is estimated as the sum of the squared distances of the query subvectors to the corresponding subquantizer centroids indexed by the fine codes.

NN search with LOPQ is highly scalable and has excellent properties in terms of both index storage requirements and query-time latencies when implemented well.

## References

More information and performance benchmarks can be found at http://image.ntua.gr/iva/research/lopq/.

Python's code implementation can be found at https://github.com/yahoo/lopq.

1. Y. Kalantidis, Y. Avrithis. [Locally Optimized Product Quantization for Approximate Nearest Neighbor Search.](http://image.ntua.gr/iva/files/lopq.pdf) CVPR 2014.
2. H. Jegou, M. Douze, and C. Schmid. [Product quantization for nearest neighbor search.](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf) PAMI, 33(1), 2011.
3. A. Babenko and V. Lempitsky. [The inverted multi-index.](http://cache-ash04.cdn.yandex.net/download.yandex.ru/company/cvpr2012.pdf) CVPR 2012.

## License

Code licensed under the Apache License, Version 2.0 license. See LICENSE file for terms.
