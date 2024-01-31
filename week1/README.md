# C1-Project-Week 1 Content Based Image Retrieval

The code can be found in the notebook called _C1_Project.ipynb_. In order to keep the code clean, the implementation of the functions we used is in _utils_/_utils.py_.

## Setting up the environment

Before we start the project development, we need to install the Python programming language. Before installing Python, it's a good idea  to check if it's already installed on our computers. We can check that by opening the terminal and running the command: 

 ```python
python --version
```
  If Python is installed, you will see the version number. If it is not, we can download it from
  the website https://www.python.org/downloads/, download it and start the installation process.
  
  After we finish that, we can download and install pip. Pip is a package manager for Python
  that simplifies the process of installing, managing, and updating packages and libraries.
  To install it we can enter on the website https://bootstrap.pypa.io/get-pip.py, download it, 
  open the terminal and type
  
   ```python
  python get-pip.py
  ```

  Once we finish with the Python setup, we can move on to install some libraries, such as:
  - OpenCV : used for tasks like object detection and image manipulation
  - Matplotlib :  commonly used for data visualization
  - Skimage : provides tools for tasks like image segmentation
  - Re : search, match, and manipulate strings based on specific patterns, making it handy
for tasks like data validation and text parsing.
  - Os : provides a way to interact with the operating system, allowing you to perform various file and directory operations.
  - Pickle : allows us to save Python objects to a file and then load them back into memory.
  - Metrics: contains different metrics to evaluate machine learning models
  We can import them by typing into the terminal the following commands:
  
```python
pip install opencv-python
pip install numpy
pip install scikit-image
pip install matplotlib
pip install regex
pip install os-sys
pip install pickle5
pip install ml_metrics
```

  Now that we have installed them, we can import them by using
  
```python
from skimage import io
import matplotlib.pyplot as plt
import cv2
import re
import pickle
import os
import ml_metrics
```

## Task 1: Create Museum and query image descriptors 
  For this task we used several color histograms, such as:
  - Gray Level Histogram: Represents pixel intensity distribution in an image.
  - RGB Histogram: Shows the distribution of red, green, and blue color channels.
  - HSV Histogram: Captures color information as Hue, Saturation, and Value components.
  - CieLab Histogram: Represents color as three values: L* for perceptual lightness and a* and b* for red, green, blue and yellow.

After trying all the color histograms, we came to the conclusion that HSV and CieLab provided the best results. They separate color from brightness, making them robust to lighting changes. They are also perceptually uniform and less sensitive to noise, aligning better with human perception. 

## Task 2 & 3 : Similarity measures

In this part we had to use some similarity measured to and for each image in QS1, compute similarities to museum images. We have implemented these 5 methods:
- Euclidian distance : Measures the straight-line distance between two points. It works well when the data is continuous and has a Gaussian distribution. However, it can be sensitive to outliers.
```math
\text{Euclidean Distance} = \sqrt{\sum_{i=1}^{n}(h_1(i) - h_2(i))^2}
```
- L1 Distance: Measures the absolute differences between the coordinates of two points. It's robust to outliers and can work well when data is not normally distributed.
```math
\text{L1 Distance} = \sum_{i=1}^{n}|h_1(i) - h_2(i)|
```
  
- χ2 Distance: Specifically designed for comparing histograms. It computes the sum of squared differences between the observed and expected values in the histograms. It's a good choice when comparing histograms.
```math
\text{χ² Distance} = \sum_{i=1}^{n} \frac{(h_1(i) - h_2(i))^2}{h_1(i) + h_2(i)}
```

- Histogram Intersection (Similarity): Measures the overlap between two histograms. It's used when you want to find how much two histograms overlap. It's a good choice when histograms represent similar data distributions.
```math
\text{Histogram Intersection (Similarity)} = \sum_{i=1}^{n} \min(h_1(i), h_2(i))
```

- Hellinger Kernel (Similarity): Also used for comparing histograms. It's based on the Hellinger distance, which measures the similarity between two probability distributions. It's useful when you want to capture the similarity between two histograms.
```math
\text{Hellinger Kernel Distance} = \sum_{i=1}^{n}(\sqrt{h_1(i) * h_2(i)})  
```

The implementations of the similarity search methods can be found in the _utils_/_utils.py_ file (for clarity, we have omitted the different combinations of various color spaces and similarity search methods that were deemed unsatisfactory from the final version of the notebook). After evaluating the results, we came to the conclusion that the best combinations were CieLab and HSV, both with Histogram intersection.

## Task 4 : Evaluation

QS1 | map@1 | map@5
| :--- | ---: | :---:
Method 1   | 50% | 58.4%
Method 2   | 83% | 85.9%

QS2 | map@1 | map@5
| :--- | ---: | :---:
Method 1   | 13.3% | 13.3%
Method 2   | 30% | 36%

## Task 5 & 6 : Remove the background and evaluate 

For the Query Set 2, we had to evaluate the mask using:
- Precision: Measures how many of the predicted positive instances were actually correct.
```math

\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
```
- Recall: It measures how many of the actual positive instances were correctly predicted
```math
\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
```

- F1-Measure: The F1-Measure is the harmonic mean of precision and recall. It provides a single score that balances both precision and recall. It's particularly useful when you want to find a balance between avoiding false positives and false negatives. It ranges from 0 to 1, where a higher value indicates a better balance between precision and recall.
```math
\text{F1-Measure} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
```
In summary, precision focuses on the accuracy of positive predictions, recall emphasizes the ability to capture all positive instances, and the F1-Measure combines both to provide a single metric for evaluating a classification model's performance.

### Results for the validation query set 2

 Precision | Recall | F1 - measure
| :--- | ---: | :---:
  0.801 | 0.960 | 0.863



### MAP@k results for the validation query set 2
Methods | map@1 | map@5
| :--- | ---: | :---:
Method 1   | 23.3% | 25.1%
Method 2   | 33.3% | 37.6%
