## The Source Code of MFCIS
---
##### If you have any problem, please contact us!
### Environment requirement
> - The computation of Persistence Diagram (PD) require the python package ![HomCloud](https://www.wpi-aimr.tohoku.ac.jp/hiraoka_labo/homcloud/index.en.html])
    and ![Dipha](https://github.com/DIPHA/dipha).
> - The code should work on the Linux operating system (The Ubuntu(>16.0.4 LTS) is recommend).
    
##### The other required python packages 
> - Python3
> - Scikit-image, Scikit-learn
> - Opencv
> - Tensorflow 1.10, keras 2.1.0
> - ...
 
 The complete list of the required python packages and their version information can be found at _requirement.txt_
 
#### The configuration of Homcloud and Dipha 
> we refer to the homepage of Homcloud https://www.wpi-aimr.tohoku.ac.jp/hiraoka_labo/homcloud/index.en.html for the configuration of Homcloud and Dipha.
### DockerFile
It's so cumbersome to do the configuration from scratch, so we provided a dockerfile, you can run the program in the docker container.
The dockerfile and homcloud package can be found in the dockerfile folder. We strongly recommend you adopt this way. The dipha provided was compiled on Debian system. If you want to change the base image, the dipha should be Recompiled.

### The configuration of the code
> - You should configuration the *_dipha_* path in 
  pershombox/_software_backends/software_backends.cfg
> - The main.py is the entrance of the program, you can uncomment or comment the code to validate different models.
>- The docker container should be run under a common user accout other than a root account.
