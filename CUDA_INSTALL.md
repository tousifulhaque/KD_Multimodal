## Setting local cuda to tensorflow
Download the cuda 11.8 runfile using the command below 
```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
```
Make the run file executable
```
chmod +x  cuda_11.8.0_520.61.05_linux.run
```

Execute the runfile

```
./cuda_11.8.0_520.61.05_linux.run  --silent --toolkit --toolkitpath=$HOME/cuda-11.8
```
Add the local cuda file path

```
export PATH=$HOME/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda-11.8/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

## Setting up cuDNN without sudo access 

Use the command below to see the cuDNN version cuda is using 

```
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```
Download necesary cudnn file in .tar.xz format. And extract it.


Copy neccasary files to local cuda folder.
```
cp cudnn-linux-x86_64-8.8.1.3_cuda11-archive/include/cudnn*.h $HOME/cuda-11.8/include/
cp cudnn-linux-x86_64-8.8.1.3_cuda11-archive/lib/libcudnn* $HOME/cuda-11.8/lib64/
```
Make the files executable. 

```
chmod +r $HOME/cuda-11.8/lib64/libcudnn*
```

Add the path to `~/.bashrc`

```
export CUDNN_HOME=$HOME/cuda-11.8
export LD_LIBRARY_PATH=$CUDNN_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDNN_HOME/include:$CPATH
export LIBRARY_PATH=$CUDNN_HOME/lib64:$LIBRARY_PATH
source ~/.bashrc
```