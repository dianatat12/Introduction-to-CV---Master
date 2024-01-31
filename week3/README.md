# C1-Project-Week 3

## Setting up the environment

For this week we need to import the following libraries:
  
```python
from skimage import io
import matplotlib.pyplot as plt
import cv2
import re
import pickle
import os
```
If you don't have Python installed or we don't how how to install it you can check the instructions from Week 1.

# Task 1: Filter noise with linear or non-linear filters

## Neighbourhood operations
Let's first start with some theorical concepts. 

Let $f(l,c)$ be the initial input image, $g(l,c)$ be the processed output image and $T$ be an operator on $f$, defined over some neighborhood of pixel $(l,c)$. The new value of any pixel within the processed image is obtained by combining  some pixel values from the initial image, placed around (in the neighborhood of) the current processed pixel. The center of the neighborhood is moved from pixel to pixel spanning the whole image. The operator $T$ is a function that combines the values extracted from the image. $T$ is applied at each location $(l,c)$ to yield the output, $g(l,c)$, at that location. The process utilizes only the pixels in the area of the image spanned by the neighborhood.

In order to define the transform, the <b>shape of the neighbourhood</b> of the current processed pixel $V_{(l,c)}$ and the <b>operator $T$</b> need to be defined.

Defining the neighborhood means specifying the locations of pixels belonging to that neighborhood, with respect to the current location, i.e. the current processed pixel located at $(l,c)$. The neighborhood $V_{(l,c)}$ is:
$$V_{(l,c)} = \{(m_1,n_1),(m_2,n_2),...,(m_k,n_k)\}$$
where $(m_i,n_i)$ are the relative coordinates with respect to the current pixel $(l,c)$. The current processed location $(l,c)$ is the origin of the coordinate system attached to the neighborhood.

In image coordinates (considering the origin as the origin of the image) the neighborhood can be written as:
$$V_{(l,c)} = \{(l+m_1,c+n_1),(l+m_2,c+n_2),...,(l+m_k,c+n_k)\}$$
and the neibhourhood operation becomes:
$$g(l,c) = T(f(l+m_1,c+n_1),f(l+m_2,c+n_2),...,f(l+m_k,c+n_k))$$

## Linear filtering of images
If $T$ is a linear function, then the transform is called <b>linear filtering</b> and is written as:
$$g(l,c) = \sum_{(m,n)\epsilon V}w_{mn}f(m+l,n+c)$$
where $w_{mn}$ are scalar constants called the <b>coefficients (or weights)</b> of the filter.

<b>Taking this into account, linear filtering is actually a weighted sum of the pixels found in the neighbourhood of the current pixel.</b>

Depending on the chosen weights, the result of the filter will differ. To simplify things, when defining a linear filter, we will use a matrix (called the <b>mask</b>) of the form of the neighbourhood that contains the coefficients of the filter. 

## Linear smoothing filters
Smoothing filters are used to increase the uniformity of the pixels in a region, thus decreasing small variations, which can be caused by noise. In that sense, smoothing filters are used to decrease (certain types of) noise in images.
The most common type of noise is the Gaussian, white, additive noise, independent to the image pixels. Usually for this type of noise the mean is considered to be 0. 

## Nonlinear filters. Rank-order filters

For different types of noise, linear filtering is not adequate. A different kind of noise is the <i>impulsive</i> noise (also called "salt and pepper" noise). This kind of noise affects only some pixels in the image, but completely corrupts the values of the pixels, by making them black (0) or white(1 or 255, depending if the image is scaled or not).

In this case, the use of an average filter would only spread the noise to other pixels. For problems like these, a class of nonlinear filters called rank-order filters are best suited to solve the issue. This class of filters does not combine the values in the neighbourhood, instead selecting just one of the values in the neighbourhood using a certain criterion.

## Methods approached
Median Filter 

    We tested this filter using 3x3 and 5x5 neighbourhood,  V[min] - V[max] and V[max] - V[min].
    We iterate through the input image, extract the local neighbourhood, sort the pixel values, and assigns 
    the median value to the corresponding pixel in the filtered image. 
    For the V[min] - V[max] approach we calculate the minimum and maximum pixel values within each neighbourhood, 
    and assign the difference between the maximum and minimum values to the corresponding pixel in the filtered 
    image. This process effectively emphasises areas of local contrast in the image. For V[max] - V[min] we did 
    the reverse operation.For this filter the 3x3 neighbourhood worked the best and it is also one of the best 
    from all methods we tested.

Rank Order Filter

    Rank order filters are filters used to emphasize or de-emphasize pixel values based on their position 
    in a sorted neighbourhood. For each pixel in the image, we extract a local neighbourhood, sorts the pixel 
    values in ascending order, and assigns the value at the specified rank within the sorted neighbourhood 
    to the corresponding pixel in the filtered image.

Arithmetic Mean Filter


    Arithmetic mean filters smooths or blurs an image by replacing each pixel's value with the average 
    value of its neighbourhood pixels. We applied an arithmetic mean filter to an input image using a 
    custom 3x3 kernel. Then we loaded the image and splits it into its colour channels. After that, we 
    create instances for each channel and applies the filter independently to each one. 

Average Filter

    For this filter we replaced each pixel in an image with the average value of the pixel and 
    its neighbouring pixels within a specified kernel.

Gaussian Filter

    Gaussian blur smooths and reduces noise in an image by applying a Gaussian filter. We replace each pixel 
    value with a weighted average of its neighbouring pixels, with more weight assigned to pixels closer to 
    the center. For this filter we have used 5x5 kernel and sigma = 3.5.

Gradient Filter 

    With this filter we calculate the gradient, which represents the rate of change in pixel values,
    at each pixel location. The computation takes place both the horizontal and vertical directions using 
    convolution masks.

Low Pass Filter

    Low-pass filters reduce high-frequency components while preserving low-frequency information. 
    These filters work by attenuating or eliminating pixel values that change rapidly in an image, 
    resulting in a smoother and less noisy image. 
    We first transform the input image into the frequency domain using the FFT . Then we create a mask 
    that defines the frequency components to keep and those to remove. After applying the mask to the 
    Fourier Transform, it performs an inverse Fourier Transform to obtain the filtered image. 

Average Filter

    For this filter we replaced each pixel in an image with the average value of the pixel and its 
    neighbouring pixels within a specified kernel.

Gaussian Filter

    Gaussian blur smooths and reduces noise in an image by applying a Gaussian filter. We replace each 
    pixel value with a weighted average of its neighbouring pixels, with more weight assigned to pixels 
    closer to the center. For this filter we have used 5x5 kernel and sigma = 3.5.

Gradient Filter 

    With this filter we calculate the gradient, which represents the rate of change in pixel values, 
    at each pixel location. The computation takes place both the horizontal and vertical directions using 
    convolution masks.

Low Pass Filter

    Low-pass filters reduce high-frequency components while preserving low-frequency information. These 
    filters work by attenuating or eliminating pixel values that change rapidly in an image, resulting 
    in a smoother and less noisy image. 
    We first transform the input image into the frequency domain using the FFT . Then we create a mask that 
    defines the frequency components to keep and those to remove. After applying the mask to the Fourier Transform,
    it performs an inverse Fourier Transform to obtain the filtered image. 

Our results are:

| Metric                      | map@1 | map@5 | Average PSNR |
|-----------------------------|-------|-------|--------------|
| Non-augmented images (upper bound) | 0.266 | 0.445 | - |
| No filter (lower bound for the noisy images) | 0.133 | 0.277 | 30.53 |
| Median filter               | 0.166 | 0.305 | 35.12 |
| Max rank order filter       | 0.133 | 0.277 | 33.94 |
| Weighted rank order filter  | 0.200 | 0.318 | 32.36 |
| Average filter              | 0.133 | 0.229 | 32.49 |
| Wavelets (Daubechies)       | 0.166 | 0.306 | 35.64 |


# Task 2: Detect box with overlapping text

In this part we focus on text extraction from denoised images and its subsequent analysis. We performed OCR on the images by applying last week's text-finding function, binarizing the section, and using pytesseract for text extraction. 

Pytesseract is used to extract text from these images, and the extracted text is stored in a dictionary named 'text_deno' where image IDs serve as keys, and the corresponding extracted text serves as values. We have also included functions for identifying text within images, such as 'find_text', which processes images to locate and define bounding boxes around detected text. The 'read_text' function then extracts text from these bounding boxes, binarizes it, and employs OCR for text conversion. Additionally, we import and processes text information from text files stored in a specified directory through the 'import_text' function. The results are collected in the 'text_deno' dictionary which we print to see the results. 

For comprehensive comparison, we conducted image retrievals using text-only data and various distance metrics, considering noisy, denoised, and non-augmented images. In the case of denoised images, wavelet-based denoising yielded the best outcomes, consistent with previous results.

For the evaluation we have used various text distance metrics for comparing the text, such as Levenshtein distance, Damerau-Levenshtein distance, Jaro distance, Needleman-Wunsch distance, Gotoh distance, MLIPNS distance, and Hamming distance with padding. We have obtained the best results with the Levenshtein distance.

For this task we have implemented the code in the "text.py" file which contains 2 functions.

    In the "TextFinderWeek2" class is we are finding text in an input image by detecting bounding boxes. 
    We are using color thresholding to segment text regions and applies various image processing techniques 
    to isolate and extract the text. The result is a set of coordinates representing the bounding box of the 
    detected text.

    "TextReader" class is designed to read and extract the text within a given bounding box. It takes the 
    image and bounding box coordinates as input, preprocesses the region, performs OCR 
    using the Tesseract library, and returns the extracted text after some text cleaning.

# Task 3: Implement texture descriptors

In this task, we streamlined our approach by choosing DCT coefficients as the texture descriptor, while recognizing the potential for further exploration. We experimented with various distance metrics, ultimately opting for L2 due to execution time considerations, although histogram intersection showed promise. We determined that 2x2 blocks and 100 coefficients struck the right balance, as increasing coefficients yielded diminishing returns, and using only one block per image produced surprisingly good results. Thus, for tasks 3 and 4, we settled on a straightforward method: 1x1 blocks, 100 coefficients, and the L2 metric, ensuring efficiency and effectiveness for our specific objectives.

    The code is in the class named "TextureDescriptor" and a subclass "DctDescriptor". The "DctDescriptor" 
    class has methods to divide an input image into blocks, perform DCT on these blocks, and extract DCT 
    coefficients. It utilizes a zigzag pattern to scan the DCT coefficients and allows you to specify the 
    number of coefficients to retain. Ultimately, it provides a feature vector that represents the texture 
    descriptor of the input image. The "calculate" method within the "DctDescriptor" class takes an image 
    as input and returns this texture descriptor. This code can be used to extract texture features from 
    images for various computer vision and image processing applications.


# Task 4: Combine descriptors

In our approach to consolidate all three descriptors into a unified one, we computed a weighted average of their retrievals, with the parameters set as follows: TT = 0.5 for texture descriptors, T = 0.2 for text descriptors, and C = 0.3 for color descriptors. This amalgamation encompasses Retrieval for texture descriptors (TT), Retrieval for text descriptors (T), and Retrieval for color descriptors (C). Our earlier results clearly demonstrated the superiority of texture descriptors, prompting us to assign them the highest weight. Subsequent slides showcase our quest to identify the most effective descriptor combination, with initial emphasis on texture (TT) due to its outstanding performance, a moderate consideration for color (C), and minimal weight for text (T) given its subpar results.

    For implementing this task we define a class named "Descriptor" and three subclasses: "TextDescriptor,
    " "HistogramDescriptor," and "TextureDescriptor." These classes are used for calculating and measuring 
    the similarity between images based on different descriptors. The "Descriptor" class provides a general 
    framework for calculating and comparing image descriptors. It includes methods for calculating query and 
    database (BBDD) descriptors and for measuring the similarity between them. The subclasses tailor the descriptor 
    calculation process to specific types of descriptors: text, histogram, and texture. Each subclass implements 
    methods to calculate descriptors for queries and the database, taking into account the characteristics of 
    their respective descriptors. The code is structured to support the comparison of various image descriptors, 
    enabling the evaluation of different similarity measures for retrieval tasks.


