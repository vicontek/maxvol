mkdir -p mnist
cd mnist
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz  && gzip --decompress train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz  && gzip --decompress train-labels-idx1-ubyte.gz

wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz && gzip --decompresst   10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz && gzip --decompress t10k-labels-idx1-ubyte.gz 

