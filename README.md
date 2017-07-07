# Sequence-Test

Here you can find some test scripts to segment a certain part over an image sequence (with an initialization)

## The different files

A few scripts are present in this repository :
- DoP_histograms : script to evaluate visually (with histograms) the consistency (in value) of DoP images for thresholding
- blob_test : a test script to evaluate the performance of a blob detector for tracking a certain part (does not give godd results)
- featurepoint_matching : Test script of an estimation of homogeneous matrix between 2 images
- regiongrowing : Region growing algorithm for segmentation
- seqtoolbox : some functions useful for the final treatment
- test3d : script to visualize with a 3d plot a gradient estimation
- test_hog : test script to evaluate the performances of HOG to solve the problem
- test_sequence : main test file. Implements the latest solution found.

## Notes and remarks

- To track the region of interest, an algorithm based on mimimum finding is used. It does not use any kind of classifier or feature detection.
- Since the "tracking" does not use a classification system, it is compulsary to initialize on the first image of the sequence.
- As it is, cannot work in real time with full resolution. Although it can be using unsupervised algorithm (such as Super Pixel or Felzenszwalb algorithm) and image decimation.
- The tracking works best with AoP images





