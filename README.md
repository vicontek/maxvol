## Motivation
In machine learning we often don't have powerful enough computers or large enough memory to perform calculations on the whole dataset which has too many features and/or samples. A dataset is represented with a matrix. If we can find a good subset of features, we'll be able to train the machine learning algorithm using computational resources. We can also try to achieve the same goal by using only the most representative samples. Feature selection and samples selection is equivalent to choosing a submatrix. For both purposes we can try choosing a submatrix with large volume, where volume as the absolute value of determinant for square matrices, is generalized for rectangular matrices. 

## Experiments
We made 4 main experiments on different datasets:
* ARCENE
* MNIST
* Housing prices
* Synthetic dataset

More details can be found in `maxvol_report.pdf`

## Team
The project was performed by the team of four people:
* Philip Blagoveschensky **@philip-bl**
* Ivan Golovatskikh **@vicontek**
* Maria Sindeeva **@lapsya**
* Mirfarid Musavian **@mirfaridmusavian**

## Prerequites
In addition to the common numerical and ML packages shuch as `scipy` and 'sklearn', you need to install `maxvolpy` package:

``pip3 install maxvolpy``
