# Building TensorFlow with Device-Annotation Support

This document outlines the steps to build TensorFlow with device annotation functionality.

1. Clone the repository from `git@github.com:WingsOfPanda/tensorflow.git`.
2. Switch to the branch `feature/device-annotation`.
3. Follow the instructions at [TensorFlow Source Installation](https://www.tensorflow.org/install/source) to build TensorFlow based on this branch.

Once you have successfully built and installed this version of TensorFlow, you can verify the installation. Typing `tf.__version__` in the Python environment should display: `"2.9.0-device-annotation-mkl-support"`.

The device-annotation feature enables users to control the placement of gradient calculations on specific devices. With this functionality, we can distribute different aspects of training (such as forward inference, gradient calculation, and backpropagation) across multiple devices, thereby mitigating Out-Of-Memory (OOM) issues.

For a practical example, refer to `demo.py`. It provides a Minimal Working Example (MWE) demonstrating how to use the device-annotation feature to distribute training tasks.
