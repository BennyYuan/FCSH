# FCSH 
No.1 Deep Learning to Full-Convolution Semantic Hash Algorithm

Tensorflow implementation for FCSH
- running equipment: Ubuntu 16.04.5 and GeForce RTX 2080 Ti

- running environment: Tensorflow 1.8.0 and python 3.6.3

- please run several times and take the average (as the result is a little bit unstable)

- you can download the [ImageNet](https://github.com/thuml/HashNet/tree/master/caffe) and [cifar-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset  

- please extract the file in /home/${user}/work/data/${dataset_name}

### Installation (sufficient for the demo)

1. Clone the FCSH repository
  ```Shell
  # Make sure to clone with --recursive
  mkdir -vp /home/${user}/work/${your_name}/FCSH
  git clone --recursive https://github.com/BennyYuan/FCSH.git
  ```

2. Download the Inception V3 checkpoint:
  ```Shell  
  mkdir -vp /home/${user}/work/data/checkpoints
  cd /home/${user}/work/data/checkpoints
  wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
  tar -xvf vgg_16_2016_08_28.tar.gz
  rm vgg_16_2016_08_28.tar.gz
  ```
    
3. Download the dataset:
  ```Shell
  mkdir -vp /home/${user}/work/data/cifar10
  cd /home/${user}/work/data/cifar10
  wget http://www.cs.toronto.edu/~kriz/cifar.html
  ```
   
       
4. Running model:
  ```Shell
  mkdir -vp /home/${user}/work/data/cifar10
  cd /home/${user}/work/${your_name}/FCSH/sh
  ./cifar10_hash_bit16_on_vgg_16.sh
  ```
    


