# building the tensorflow with device-annotation

We describe the step of building the tensorflow with device annotation functionality.

- clone the repo from git@github.com:WingsOfPanda/tensorflow.git
- shift to branch 'feature/device-anotation'
- following the instruction from [url][https://www.tensorflow.org/install/source] to build the tensorflow based on this branch


After successfully built and installed this tensorflow, type 
'tf.__version__' you should have:
"2.9.0-device-annotation-mkl-support"

The device-annotation allows user to control the placement of gradients calculation on the spcific device. This, we can split the training, including forward inference, gradients calculation, and backpropagations, onto different devices to avoid OOM issues. 

The demo.py gives a MWE of how to use device-annotation to split training.