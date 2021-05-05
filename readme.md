## The Source Code of MFCIS 

Clicking the follow image to access the website.

![MFCIS online platform](https://github.com/WeizhenLiuBioinform/mfcis/banner.png)
---
##### If you have any problem, please contact us!
### ** If the CUDA installation process failed. Please remove the installation commands of cuda and enter the docker container and install the CUDA manually. Please make sure tensorflow and CUDA versions are compatible.
### Environment requirement
> - The computation of Persistence Diagram (PD) require the python package ![HomCloud](https://homcloud.dev/])
    and ![Dipha](https://github.com/DIPHA/dipha).
> - The code should work on the Linux operating system (The Ubuntu(>16.0.4 LTS) is recommend).

##### The other required python packages 
> - Python3
> - Scikit-image, Scikit-learn
> - Opencv
> - Tensorflow > 2.0 
> - ...

 The complete list of the required python packages and their version information can be found at _requirement.txt_

#### The configuration of Homcloud and Dipha 
> we refer to the homepage of Homcloud https://www.wpi-aimr.tohoku.ac.jp/hiraoka_labo/homcloud/index.en.html for the configuration of Homcloud and Dipha.
### DockerFile
It's so cumbersome to do the configuration from scratch, so we provided a dockerfile, you can run the program in the docker container.
The dockerfile and homcloud package can be found in the dockerfile folder. We strongly recommend you adopt this way. The dipha provided was compiled on Debian system. If you want to change the base image, the dipha should be Recompiled.
### Tips:
 - You can enter the dockerfile folder and execute the following command:
 1. Building the docker image
 > docker build -t mfcis:1.0 ./
 2. Creating the docker container. If you have built the docker image successfully, please create and start the docker container.
 > docker run -it --name mfcis -p 2202:22 -v your_local_code_path:/workspace --gpus all --shm-size="32g" mfcis:1.0 /bin/bash
 - your_local_path is the real path of the code and dataset on your computer. Please ensure all the path are properly configured before running the code.
 3. The docker container create the root user as default, but the dipha should work under the common user, as a result you can create a new user account in the docker container and run the feature_extraction program on this account and train and test the model on the root account. It's a little difficult to configure the runing enviroment, so please contact us or raise an issue. we will reply you as soon as possible.
 ##### The dockerfile and configuration has been tested on a linux server with NVIDIA TESLA v100 and GTX1080TI.
### The configuration of the code
> - You should configuration the *_dipha_* path in 
  pershombox/_software_backends/software_backends.cfg
> - The main.py is the entrance of the program, you can uncomment or comment the code to validate different models.
>- The docker container should be run under a common user accout other than a root account.

### The detail configuration of file path and other settings are shown in the source code by comment.
The core functions are under the folder named feature_extraction and model, you can also modify the pipeline in your way.
