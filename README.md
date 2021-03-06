![alt text](https://github.com/julianzimmerlin/pgm-project/blob/master/sample.png?raw=true)
<br/>
PGM Project
This is the code for the practical project of the Probabilistic Graphical Models class 2021 at Saarland University.
In this project, we address the problem of color image denoising.

To reproduce the results given in tables one and two, open the file baseline_evaluation.m. In the file, load the filters and alphas that you want to use (e.g. filters_200_500.mat and alphas_200_500.mat).
Then execute the script. For denoising in the YCbCr color space, you additionally need to uncomment the lines that convert the noisy images to YCbCr and back and open the file denoise_foe.m and uncomment the marked line there. Keep in mind that you still need to pass the original RGB image to denoise_foe for correct PSNR calculation.
Also, in all cases, make sure the hyperparameters are as described in the report.

The reproduce the table from the YCbCr discussion section, execute the file ycbcr_vs_rgb.m and also uncomment the marked line in denoise_foe.m

To learn filters and alphas, use the file learnFilters.m, which internally calls learnAlphas.mat. To change the number of samples, you need to change the variable num_samples in both files.

The inference folder contains the inference code for McAuley's method (taken from his website), changed slightly to return the peak PSNR and image and make YCbCr denoising possible.

The inference_baseline folder contains inference code for the baseline, which is a modification of assignment 3 for color images.

The DCGAN folder contains the PyTorch implementation of the GAN that we used to augment the pool of filters.

You also need to download the BSDS500 dataset and add it to the directory if you want to evaluate the code on the test images.
To train the DCGAN, go the *DCGAN* folder and run
`python train.py`
To generate the filters using DCGAN, go to the *DCGAN* folder and run
`python generateFilters.py `

