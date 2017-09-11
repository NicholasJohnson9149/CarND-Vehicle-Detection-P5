## Vehicle Detection Writeup 
#### By Nicholas Johnson 
#### September 7th 2017 

This writeup will explain the steps required to create a vehicle detection pipeline and look at ways to improve the system for future revisions.

---

**Vehicle Detection Project**

<!-- The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected. -->

[//]: # (Image References)
[image1]: ./output_images/HOG-feature-array.png "Picture of both cars and notcars HOG features"
[image2]: ./output_images/windows-profile.png "Final window layout for pipeline "
[image3]: ./output_images/bad_window_profile.png "One of the test I tried with windows that failed"
[image4]: ./output_images/false_positives.png  "First pass, shows false positives, minimal tweaking of perimeters"
[image5]: ./output_images/false_positives2.png  "Change from RGB to YCrCb and got less false positives"
[image6]: ./output_images/few-false-positives.png "Changed window layout and size along with YCrCb to LUV and have few to no false positives"
[image7]: ./output_images/false_positives_heatmap.png "heatmap with no threshold "
[image8]: ./output_images/Heatmap_imgaes.png "heatmap with a threshold of 10"
[image9]: ./output_images/complete_pipeline_vistralization.png "A complete pipeline visualization"
[image10]: ./output_images/Last-detection-of-video.png "last detection of project video, screen capture from video"
[video1]: ./output_videos/project_clip_out.mp4 "Final Video with smoothest output boxes"

<!-- ## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! -->


###Histogram of Oriented Gradients (HOG)

####1. I extract the HOT features at two places in my code, first to visually inspect the HOG features on training images and then using the `get_hog_features()` function. Tish function is used a few times in the pipeline and during evaluations of the data. I declare all my functions in the first section of my pipeline and this function is to first one of that. All my work is down in a jupyter notebook so I will not use line numbers in this writeup as they might change as I make edited and also because notebooks don't have line numbers. 

I will refer to sections in the notebook and some code in my VehicleDetections.py using line number only when truly necessary to get my point across. 

The code for this step is contained in the fifth code cell of the IPython notebook. This cell looks at 5 random images from both the cars and notcars data set we train on latter. Ideally 

I started by reading in all the `vehicle` and `non-vehicle` images. I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2) of the output of five car and 5 not car images from the data set. 

![alt text][image1]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters passed off of what i saw used in the walk-through lecture and also the lectures have a great code example to play with the HOG perimeters. I then created a section in my code to play with just one set of permeates to, I Wikied HOG and found it stands for Histogram of oriented gradients. Once I understood what the HOG function was doing in the sklearn library I relied that "ALL" channels is the best option because  running any channel individually didn't works as well. It does not give all the details between changes in gradient for each layer. The technique counts occurrences of gradient orientation in localized portions of an image. This method is similar to that of edge orientation histograms; similar to Canny edge detection used in the first project; but differs in that it is computed on a dense grid of uniformly spaced cells and uses overlapping local contrast normalization for improved accuracy.  

So next we chose the orient number to be 9, this is because in lecture it was said that you get no real gain after 9 and anything below will not have as great of effect on the HOG functions output. This is then covered in grater detail in a 

``` python
	# define features parameters 
    color_space = 'YUV' #can be RGB, HSV, LUV, HLS, YUV. YCrCb 
    orient = 9 
    pix_per_cell = 8
    cell_per_block = 2 
    hog_channel = 2 # Can be 0,1, 2, or "ALL"
    spatial_size = (32, 32) #spatial binning dimensions 
    hist_bins = 32 # number of histogram bins 
    spatial_feat = True # spatial features on or off
    hist_feat = True # histogram features on or off 
    hog_feat = True # HOG features on or off 
```

YUV is a color space typically used as part of a color image pipeline. It encodes a color image or video taking human perception into account, allowing reduced bandwidth for chrominance components, thereby typically enabling transmission errors or compression artifacts to be more efficiently masked by the human perception than using a "direct" RGB-representation. Other color spaces have similar properties, and the main reason to implement or investigate properties of Y′UV would be for interfacing with analog or digital television or photographic equipment that conforms to certain YUV standards. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features.

I trained a linear SVM because it was recommended in the lectures and appears to work well. I did look at the `sklearn.svm.linearSVC` documentation, <a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html">HERE</a>, to figureout what perimeters I could use. I ended up using the linear SVN because it is really just a Linear Support Vector Classification network designed to find traits on images and video. I read through a few other options and even tried the `svn.linearSVR` but foud it preformed poorly, taking longer to train and not achieve smooth results.

I did play with the perimeters during training the `svc = LinearSVC(random_state=0, loss='hinge')` to make sure I would get consitent results. When I first trained the model I got 94% which is terrible when you consider the number of samples I take per image. I played around with some code from the walkthrough to understand the impact of windows on performance and also dealing with false positives. 

Two important points to getting better training results, where to normalize your training data by use sklearn.preprocessing.StandardScaler() to extract feature vectors for training your classifier as described in this lesson. Then apply the same scaling to each of the feature vectors you extract from windows in your test images.I did this using the `extract_features()` functions and then create a standard scaler. Which ended up causing issues in one of my models latter  due to and issue with teh data being overly randomized. 

See one of the recommendations is to random shuffling the data, and by doing this you can create new data sets to train and validate on without increasing the chance of over-fitting. 

``` python 
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
```

I used the code above in all my notebooks to break the image data into training and test sets.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I ended up using multiple windows to find images at different distances to the camera. I created a function like that shown in lectures to cycle through start and stopping point on x and y axis of an image. I then used this to cycle through and image to and look for vehicles. This method would take the HOG for each iteration slowing down the time to processes each image, or in our final case vied frame. In the walkthrough video the instructor creates a find cars function that takes the HOG of the whole image at once and then cycles through looking for car features based on the model. This method saves computational time and while it might generate more false positives can be fixed in latter steps using a heatmap threshold. 

The number of false positives was more than I would have liked so I went back to the model and tweaked the spiatail size and hist_bins to create less noise in the model. by having them larger than 16 i was able to create HOG samples large enough to limit false positives in some of the different lighting conditions but still not perfect. 

The function can be found in the second cell at the bottom of the Jupiter notebook, it is basically a direct copy of what is used in class. I use the sliding window method only once when evaluating the test image, it's also used when training the model. 

I then played with overlap and size of the windows to find features of cars using the HOG features explained above. After a few interaction I found that 80% overlap and limiting the area the windows looked helped decrease false positives and also decrease processing time of each image. 

You can see in the notebook a output showing what I finally settled on, as it gave me the best results for the lowest number of windows. By cropping in both the X and Y axis of the image I was able to just focus the search on the road. I also change the size of the windows at different distances from the camera to account for cars being smaller at a distance and larger near the camera. The image here shows the windows used in the final pipeline.   

![alt text][image2]

I did try different layouts of the windows on the image to see how it affected detection, the one that I found was worse was splitting different size windows with no overlap like shown bellow. 

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

Here is the first pass images with a 98% accuracy after training, it shows many false positives, often in shadows. I sued the YCrCb and (16,16) spacial features  with 16 histogram bins for the HOG function. 

![alt text][image5]

Next I changed the spacial feature to (32,32) and the histogram bins to 32 and achieved the above result. This was better but not good enough to get clean smooth bounding boxes around cars. In fact I ran the video on it and has so much jitter around cars with a few false positives showing. 

![alt text][image6]

I ended up using a number of different parameters to smooth out the false positives and realize the results pictured above. After changing to colorspace to LUV the white car was detected more often. I then added a heatmap threshold that I will describe in the final section of this report to limit the number of false positives.  
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are OK as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result]("https://youtu.be/EwEaZK-8Ij0") or ![alt text][video1]

I end-up with so many video clips while training, it's amazing how different values in any of the parameters can have noticeable effects on the jitter of bounding boxes around detected cars. 

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video using a class for called `vehicleprocessing`. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.   

Here's an example result showing the heatmap from a series of frames of video, and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image7]

![alt text][image8]

Creating a heatmap threshold in my class had a large impact on removing false positives, I found that 5 detection using my pipeline was enough to have a high level of confidence in that detection. i then created a function inside that class that counted the number of times a detection was found in the same area or close to the same area as in previous frames. I then took a rolling average of this value and only drew a box around what had more than 5 detection in those pixels. 


### Here is the output of of my entire model from start to finish, you can see the number of times the pipeline detected cars in a image and then the heatmap from those detections. Then the scipy.ndimage.measurements.label() from some number of images (frames). This gives a nice area to frame with a box. The last image is drawing boxes around area of the picture with the highest confidence of a car. by taking the walk troughs advice and creating a class I was able to store detection from frame to frame and predict with some level of accuracy where the car would be next. 

![alt text][image9]

### Here the resulting bounding boxes  drawn onto the last frame in the project video. As you can see by using the class to create a heatmap not the find_cars function I'm able to add smoothing to output. This helped eliminate a lot of jitter I struggled to remove in previous renditions of this assignment. 

![alt text][image10]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

So I really wanted to use this for roadside vehicle detection, I think the best way to make this more robust is to use a more detailed model and a large set of training data. We learned in lesson one about different ways to extract features, and then in lesson two how to detect specific types of objects using neural networks, first with tensor flow then again to drive with keras. So I think if I combine all these new skills together I would be able to not just detect where a car is in an image frame but what type of car and how large it is with respect to other objects in the images. To do this I would want to create a more detailed network and then use those images to train it, along with classified images of cars, trucks and motorcycles. Then I would create a vehicle class that would use some of teh things we learned in this project like HOG and color detection to create a robust model. 

This is the fun part, how to improve the pipeline, and after trying 2 different techniques I found that creating a class helped simplify the number of function I used in my first iteration. While the walk through was a great start, the method used in this project has so much room for improvement. I want to try using the LeNet architecture too help improve the speed and accuracy of this model, while also changing programing languages to C++ for faster processing. Honestly I had a really hard time with python because of the number of libraries we used in this class and my lack of familiarity with them. 

I write most of my work code in C and C++ and up until recently have not ever shipped python to customers, server side work can tolerate python, and Java script but embed systems like the ones that do a lot of this image detection in cars need to run as efficiently as possible of limited resources. 

I think that using python has been a great way to quickly learn the concepts of machine vision, neural networks, and artificial intelligence. But until it's ported or restructured for C++ it wont have the  computation robustness for real life applications.   

