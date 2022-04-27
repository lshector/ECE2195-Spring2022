This folder contains the general file structure and code used to evaluate the performance of both the
serial and accelerated CNN-JPEG algorithms. To run these files, a few additional python packages are required.
This includes tensorflow 2.8, opencv, and scikit-image. These dependencies can be installed directly into 
your conda environment via the following commands:
conda install tensorflow=2.8
conda install opencv-python
conda install scikit-image
Once these dependencies have been obtained, we can run the scripts included in this directory. Note that for 
size concerns, the data used to obtain the results included in the report has not been included in this 
repository. A short description of each script is included below:

--- CNNJPEG-encoder.py ---
This file is the baseline implementation of the CNN-JPEG encoder network. This code loads the pretrained encoder weights from the weightsComCNN.npz file and produces the saved model file Encoder-Model.hdf5. Inside
CNNJPEG-encoder.py, we have two variables, inputDir and outputDir, which correspond to the directories 
containing the original uncompressed images and the output CNNJPEG compressed images. By default, inputDir 
should point to ./inputs_val/ and outputDir should point to ./serial_val/. This code assumes that inputDir 
contains only image files, and it will attempt to compress each image within that directory and save the
results to outputDir. Running this code will also print out the total elapsed time, in seconds, that it 
takes to compress each of the test images. It will also print out the throughput as the number of images 
compressed per second. Lastly, this code also outputs the saved model Encoder-Model.hdf5, in case we 
want to load this model directly into other things.

--- calculateMetrics_x.py ---
These files are used for gathering evaluation data from both the serial and accelerated CNNJPEG 
compression. For these files to work, we want to make sure that we have images present in serial_val, 
inputs_val, and outputs_val. The images from inputs_val and outputs_val should be copied over from the 
Vitis AI implementation. To make sure that we serial_val is populated, we can run CNNJPEG-Encoder.py as
above. Once these folders are full, each of the calculateMetrics files should be able to run without 
issue. This file comes in a few different flavors, but for each of them we will calculate three main 
values that we are interested in. these files will loop through each of the images in our data set and 
calculate PSNR, SSIM, and compression ratio for each in the set. These metrics are defined in the report 
document. We separated each of these files to make gathering metrics data from each of the implementations
relatively easy.

CalculateMetrics_base.py: This file compares the output of the serial baseline version to that of the 
FPGA accelerated version. This is useful for directly determining the effects of quantization on the 
performance of the compression model, though it does not give us a sense of image quality compared to the
uncompressed images. For that, we use the other two files.

CalculateMetrics_OriginalsAccel.py: This file generates metrics between the original, uncompressed file
and the FPGA accelerated impementation version. This is useful to get an idea of how CNNJPEG compression
actually affects the images.

CalculateMetrics_OriginalsSerial.py: This is more or less the same as CalculateMetrics_OriginalAccel.py, 
but this time we use the uncompressed images and the serial baseline compressed images to generate our data.
This way, we can compare the results of this compression scheme to other compression schemes, or see how 
serial version stacks up to the accelerated version.





