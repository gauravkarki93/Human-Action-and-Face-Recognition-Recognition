# Human Action and Face Recognition

Face recognition using the L1-principal and then using similar approach for human action recognition. To calculate the L1-Principal Component for rank k, we used the individual method and matched the testing images to training images using Nearest Subspace Algorithm. L1-subspace works better than the L2-subspace under the effect of outliers. The experiments in this paper supports this and verifies as results in human action recognition where improved when we used individual method.

## Getting Started

In order to run the project you need Python3 with libraries installed: Numpy, Matplotlib, OpenCV.
The project requires high amount of matrix multiplications, so be careful not to burn your CPU/GPU.

## Human Action Recognition Algorithm:

1. Read the video for actions j = [walk, jack, bend] using
the OpenCV library in python.
2. Capture frames from the video and resize them to
60x48.
3. Consider the frame for human action j = [walk, jack,
bend], then 2D frame X = [ ]60x48
4. Calculate the absolute difference between each 4th
frame and convert it into grayscale.
5. Pass the grayscale image through a threshold into
order to remove the background surroundings from
moving object.
6. The resulting object is a motion history image which
is similar to a silhouette.
7. Flatten the resulting Motion History Image for class j
= [walk, jack, bend] and calculate the L1-Principal
Components using Individual method1.
8. Using greedy approach, find QL1
(j) = [q1
(j), q2
(j),…,qk
(j)]
for all j = [walk, jack, bend].
9. Test some random sample videos using the Nearest
Subspace Algorithm.
10. Test the Robust Database = [walking with dog,
walking with bag, ….]1x11 using Nearest Subspace
Algorithm.

## Authors

* **Gaurav Karki** - Santa Clara University

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](https://github.com/gauravkarki93/Artificial-Intelligence/blob/master/LICENSE) file for details
