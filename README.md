# PointClouds
Reading RGB-D Images of a Scene.
Create a Cloud of Points for every pixel in the image.
Find Closest neighbor with bruteforce for every 2 continuous images of the scene.
Implement KDTree to reduce computation time of closest neighbor.
Compute Optimal Translations and Rotations of the closest neightbors,minimize error, SVD Computation.
Implement Sobel Filter to use only the significant points for the Trans and Rots.
Pixelized Triangulation of the points for every different depth.
Output the total scene.
