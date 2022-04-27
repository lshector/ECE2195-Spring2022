# ECE2195-Spring2022
Artifacts Repository for ECE2195 Project

## Software Dependencies
This project depends on the Xilinx Vitis-AI toolbox. You will need docker as well. To install the toolbox, follow the instructions in the following link:
https://github.com/Xilinx/Vitis-AI-Tutorials/tree/master/Introduction/02-System_Setup

## Installation
Assuming Vitis-AI and Docker are installed, open a terminal session and execute the following commands:
```shell
cd vitis-ai-prj
./docker_run.sh
conda activate vitis-ai-tensorflow2
```

## Experiment Workflow
### (Step 0) Converting dataset images to TFRecords
First, place the training and test images under build/dataset/train and build/dataset/test, respectively. Then, convert images to TFRecords using the following command:
```shell
python -u images_to_tfrec.py 2>&1 | tee tfrec.log
```

### (Step 1) Training
For this project, we provide the pre-trained floating-point model under build/float_model/f_model.h5, so there is no need to train a model.

### (Step 2) Quantization
Quantize the model to use fixed-point arithmetic by running the following command:
```shell
python -u quantize.py --evaluate 2>&1 | tee quantize.log
```

### (Step 3) Compiling for the Target
Compile the quantized model for the ZCU104 platform using the following command:
```shell
source compile.sh zcu104
```

### (Step 4) Prepare Target Application
Copy the output files to the target by running the following command:
```shell
python -u target.py -i build/dataset/test -m build/compiled_zcu104/customcnn.xmodel -t target_zcu104 2>&1 | tee target_zcu104.log
```

Now, you will need to prepare an SD card using the following instructions:
https://github.com/Xilinx/Vitis-AI/tree/master/setup/mpsoc/VART#step2-setup-the-target

Now, you will copy the target_zcu104 folder to the ``/home/root`` folder of the SD card. Make the ``./root`` folder writeable by issuing the command ``sudo chmod -R 777`` root and then copy the entire target_zcu104 folder from the host machine into the ``/home/root`` folder of the SD card.

Alternatively, pre-built contents for the BOOT and target_zcu104 folders (excluding the test images) are available under the sd-card/ directory.

### (Step 5) Running the application on the target
Insert the SD card into the ZCU104 and connect to it via UART or an SSH connection via Ethernet. The application can be started by navigating to the target_zcu104 folder on the evaluation board and then issuing the command ``python3 app.py``. The application will start and after a few seconds will show the throughput in frames/sec, like this:

```shell
root@xilinx-zcu104-2021_1:~# cd target_zcu102/
root@xilinx-zcu104-2021_1:~/target_zcu102# python3 app.py
Command line options:
 --input_dir  :  inputs
 --output_dir :  outputs
 --threads    :  1
 --model      :  customcnn.xmodel
------------------------------------
Pre-processing 89 images...
Starting 1 threads...
------------------------------------
Throughput=65.19 fps, total frames = 89, time=1.3653 seconds
```