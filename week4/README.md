# Team4 - Week4

## Task 1
This week, we tried 3 different keypoint descriptors:

- SIFT (Scale-Invariant Feature Transform): identifies and describes local features in an image, which are invariant to scaling, rotation and illumination changes. It identifies potential interest points at different scales using the DoG method and constructs a descriptor around them.

- ORB (Oriented FAST and Rotated BRIEF): it is the fusion of FAST keypoint detector and BRIEF descriptor with many modifications to enhance the performance. It uses FAST to find keypoints, then apply Harris corner measure to find top N points among them. It also uses a pyramid to produce multiscale-features. ORB also is invariant to rotation, like presented here: https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

- FAST (Features from Accelerated Segment Test): FAST focuses on speed and efficiency by using a simple intensity comparison test on a set of pixels around a candidate pixel. It quickly identifies potential keypoints based on brightness variations, followed by non-maximal suppression to select the most distinct and stable points, but it doesn't provide descriptors for these keypoints [2].

      Difficulties encountered:
      - By computing the keypoint descriptors, we came to some difficulties that affected the
        result and retrieval of descriptors. They can be classified in 3 sections:
      - Execution time and limitations
      - Pipeline error accumulation
      - Disposition of the keypoints

  Execution time and limitations:

        The execution time for calculating the descriptors and later computing the
        distances was too long, so we had to limit the number of keypoints per picture to 3500.
  
  Pipeline error accumulation:

        Since the images are not cropped and can contain more than one painting and noise, we used
      the code we had from past weeks to crop and denoise them. The noise removal procedure worked
      fairly well. Nonetheless, the detection of the masks is not perfect. For example, the mask detection
      algorithm fails to successfully detect the painting on the image, making it impossible to extract good
      keypoint descriptors. Another example is for the image 29, that the mask detection algorithm doesn’t
      detect that there are two paintings in the picture.

  Disposition of the keypoints:

       Due to the accumulation of error from the pipeline, sometimes the keypoint detection is not
      perfect. However, sometimes even if the painting is well cropped, the keypoint detector finds
      locations that are not well located or won’t be useful for the matching. For example, in the Edvard
      Munch’s picture, there are a lot of descriptors located on the frame. Moreover, the detector locates
      a lot of descriptors around the text of the query images (like in the orange picture), however this
      shouldn’t affect the matching since the database images don’t have the text, so the descriptors won’t
      be matched.
  
        Nonetheless, in all cases, the descriptors seem to actually find the corners. Just by looking at the
      images with the descriptors, all seem to perform similarly and in many cases correctly, such as in the
      Jose Luis Pascual’s painting.

  ## Task 2

We have used the distances:

**KNN:** measures the dissimilarity between two sets of data points by considering the distances to their k-nearest neighbors, where a smaller distance indicates greater similarity.

$$\text{KNN distance}(\mathbf{a}, \mathbf{b}) = \sum\limits_{i=1}^{N}
\begin{aligned}
    1, & \quad \text{if } m_i.\text{distance} < 0.75 \cdot n_i.\text{distance} \\
    0, & \quad \text{otherwise}
\end{aligned}$$


**L1:** measures the sum of the absolute differences between corresponding elements of two vectors or sets of data.

$$\text{L1 norm}(\mathbf{a}, \mathbf{b}) = \sum\limits_{i=1}^{N} |a_i - b_i| $$

**L2:** calculates the straight-line distance between two points in a multidimensional space, providing a measure of their dissimilarity.

$$\text{L2 norm}(\mathbf{a}, \mathbf{b}) = \sqrt{\sum\limits_{i=1}^{N} (a_i - b_i)^2}$$

**FLANN:** approximate nearest neighbor search.

$$\text{FLANN distance}(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{N}
\begin{aligned}
    1, & \quad \text{if } m_i.\text{distance} < 0.7 \cdot n_i.\text{distance} \\
    0, & \quad \text{otherwise}
\end{aligned}$$

**Hamming**: quantifies the number of differing bits between two binary strings of equal length, representing the dissimilarity in their bit patterns.
$$\text{Hamming distance}(\mathbf{a}, \mathbf{b}) = \sum\limits_{i=1}^{N} \text{{bitwise XOR}}(a_i, b_i)$$

The choice to employ the ratio test, as proposed by David Lowe in the SIFT paper [1], for SIFT is motivated by the need to filter out very similar key-point matches. 

Lowe's ratio test provides a robust and algorithm-agnostic criterion by ensuring a significant difference between the distances of the best and second-best matches for each key-point, offering an effective method to discard unreliable matches and improve the accuracy of feature matching in computer vision applications.

In the code we have implemented, the ratio test is applied to the key-point matches to filter out unreliable correspondences, retaining matches where the distance to the best match m is significantly smaller than the distance to the second-best match n.

The subsequent code block sets up RANSAC by creating arrays of source  and destination  points based on the filtered key-points assign the ratio test. RANSAC is then applied using cv.findHomography to estimate a homography matrix M, and the count of inliers is determined.

## Task 3 

After performing the retrieval and evaluation using the two best methods from this week and last week (week3 and week 4 descriptors), we can observe a big difference in the results. The week 4 descriptor (keypoints) clearly outperformed the week 3 descriptor (combination of colour, text and texture).

We suspect that the results for the W3 are so low because we didn’t improve the text finder function, and so the results may be influenced by it. 

With W4 descriptor, the results are fairly successful in comparison to W3.

The transition from the week 3 descriptor, combining color, text, and texture, to the week 4 descriptor centred on key-points with RANSAC integration resulted in a substantial improvement in retrieval performance which can be observed by the increase in MAP score.


### Conclusions

- In summary, when applying key-point descriptors to non-enhanced images, the performance of the evaluated methods was consistently excellent, achieving a perfect score of 1.0. This highlights the effectiveness of key-point descriptors, especially on sharp images, where they demonstrate robustness and accuracy of feature matching.  

- The presence of masks in the image has a significant impact on performance, causing challenges such as imperfect mask detection leading to extraction problems. Additionally, the influence of noise highlights the need for enhanced preprocessing to improve the reliability of key-point descriptors in the presence of image complexity. 

- In the comparative analysis of SIFT, ORB, and FAST descriptors with and without RANSAC across week 3 and week 4 datasets, the results demonstrate varying performance. Notably, SIFT with KNN (k=2) and L1 distance consistently outperforms other configurations, showcasing robustness in both mapping accuracy and precision. The influence of RANSAC is evident in improved results, emphasizing its beneficial impact on feature matching reliability for diverse descriptor types.

- RANSAC integration improved results, especially compared to scenarios where RANSAC was not applied. By using RANSAC, the implementation  improved the robustness of feature matching, demonstrating the importance of outlier removal for accurate homogeneity estimation and thus overall performance. better. 

- The F1 measure optimisation, addressing occasional imperfections in key-point detection, enhances matching precision by setting a threshold on the number of matches. Despite variations like descriptors on frames or irrelevant text, this proactive approach ensures robustness in discarding queries not in the dataset, reflecting a strategic system refinement.

- In summary, key-point descriptors such as SIFT, ORB, and FAST are found to be effective on unenhanced images, despite difficulties in detecting masks and noise. Using RANSAC and F1 metric optimisation strategies significantly improved performance, highlighting the importance of robust preprocessing and custom strategies for real-world image matching.



    





  
