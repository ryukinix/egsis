![test](https://github.com/ryukinix/egsis/actions/workflows/test.yml/badge.svg)

# EGSIS

EGSIS is acronymoun for: Exploratory Graph-Based Semi-supervised Image
Segmentation.

It's a Python implementation of a image segmentation algorithm that
combines superpixel with complex networks dynamics. In this setup, we
classify the algorithm as transductive as well.

# Showcase on grabcut dataset

![showcase](pics/egsis-showcase.png)

First segmentation mask is the result of EGSIS segmentation over lasso
annotation from GrabCut dataset, second segmentation mask is the
ground truth.


# What is transductive segmentation?

Transductive segmentation is a concept in machine learning and
computer vision. It refers to the process of segmenting or dividing an
unlabeled dataset into distinct groups or segments based on the
inherent structure or patterns within the data.

Formally, transductive segmentation can be defined as follows:

> Given an unlabeled dataset X = {x1, x2, ..., xn}, the goal of
transductive segmentation is to assign a label yi to each data point
xi such that the resulting segmentation optimally reflects the
inherent structure or patterns within the data. This is typically
achieved by defining a similarity measure between data points and then
grouping together data points that are similar according to this
measure.

The **key characteristic** of transductive segmentation is that it does
not require a separate training phase. Instead, it directly infers the
labels for the given dataset based on the data itself. This makes it
particularly suitable for tasks where the distribution of the data is
unknown or may change over time.

# What is graph-based image segmentation?

Graph-based image segmentation algorithms are a type of computer
vision algorithm that uses a graph structure to represent an
image. The nodes of the graph represent the pixels of the image, and
the edges of the graph represent the relationships between the
pixels. The goal of graph-based image segmentation algorithms is to
partition the image into meaningful regions, such as objects or
regions of interest. These algorithms typically use a combination of
graph-theoretic techniques, such as graph cuts, minimum spanning
trees, and shortest paths, to identify the regions in the image.

In the case of this work, it uses the region as superpixels, a node it's
represented as a superpixel instead of a simple pixel. The edges are
calculated as similarity of the feature vectors between the nodes. The
main technique used to calculate the edges it's the neighbors of superpixels.

# What are superpixels?

Superpixels are a type of image segmentation technique that divides an
image into smaller, more homogeneous regions. Superpixels are
typically generated using algorithms that group pixels together based
on color, texture, and other features. The goal of superpixels is to
reduce the amount of data in an image while preserving the important
features of the image.

In the case of this work, we use SLIC, which is a simple technique as
variation of k-means algorithm considering the color space beyond the
euclidian distance.

# What are complex networks?

Complex networks are networks that contain a large number of nodes and
edges that are connected in a non-trivial way. These networks are
often used to model real-world systems such as social networks,
transportation networks, and biological networks. They are
characterized by their high degree of interconnectedness,
non-linearity, and the presence of feedback loops.

# License

BSD 3-Clause
