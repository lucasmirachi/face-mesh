# Face Mesh with MediaPipe and OpenCV
 
![Face Mesh](/images/FaceMesh.gif)

## Introduction

This is a study project I developed to create a real time Face Mesh program to better understand how to implement it in future projects.

Most of this code was taken as reference from this [informative video](https://www.youtube.com/watch?v=01sAkU_NvOY) created by [Murtaza Hassan](https://www.youtube.com/channel/UCYUjYU5FveRAscQ8V21w81A).

## Development 

For this code, it was utilized the [MediaPipe](https://google.github.io/mediapipe/) framework. MediaPipe is a framework developed by Google that contains some amazing models that allows us to quickly get started with some fundamental AI problems such as Face Detection, Object Detection, Hair Segmentation and much more!

Having said that, the model I'll be working with for this project is going to be the [Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html). Basically, it is a solution that estimates <mark>468 3D face landmarks</mark> in real-time even on mobile devices. It employs machine learning (ML) to infer the 3D facial surface, requiring only a single camera input without the need for a dedicated depth sensor. 

![Face Mesh](https://mediapipe.dev/images/mobile/face_mesh_android_gpu.gif)

[Source](https://google.github.io/mediapipe/solutions/face_mesh.html)


## Files

<mark>FaceMeshMin.py</mark> contains the minimum code that is required to run the Face Mesh program. This code can (and will!) be changed and improved on future projects.

<mark>FaceMeshModule</mark> was created because, next time I need to use a FaceMesh program inside another project, I won't need to write all of it again! By "just" importing FaceMeshModule tn the project, it is possible to make use of the algorithm!

# Next Steps

As next steps, I pretend to play with how to modify an image with this model applied to it, just like it is done on Snapchat/Instagram filters.

![](https://mediapipe.dev/images/face_mesh_ar_effects.gif)