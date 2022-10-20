# Vision Aided Navigation:
  Over the semester we implemented a system that estimates the trajectory of the vehicle from a video captured with an onboard stereo camera.
  The goal of this system is to understand the geometry of the world around us, and our position in it (Mapping and Localization).
  With high accuracy and autonomy, such technology is "life changing", a quite remarkable example would be self-driving vehicles (but not only):

  The United States Department of Transportation (USDOT) predicts that the rise of driverless cars will see the number of traffic deaths fall drastically - 90% reduction in traffic deaths.
  Higher levels of autonomy have the potential to reduce risky and dangerous driver behaviors
  (self-driving vehicles can help reduce driver error.).
  Fewer accidents mean less traffic congestion, which means a drop in emissions. But this is not just due to a reduction in accidents - 60% drop in harmful emissions.

  Those reason and many more, such as Assistive Robots for people with disabilities,
  explaining why Vision Aided Navigation is highly researched area.

  As discussed in the lectures this technology gives an answer for many other deficient  technologies, such as GPS:
  *	Robustness.
  *	Independence from the presence of infrastructure.
  *	Orientation info'.

  The following plots describes the system's process from an initial estimation of the trajectory to the final estimated trajectory after optimization (Bundle Adjustment) and loop closure:





![Initial estimation](https://user-images.githubusercontent.com/82065601/196910583-34533b7e-73ee-44e8-8ddb-0f5314618e6f.png) ![Final estimation](https://user-images.githubusercontent.com/82065601/196910745-79491e0c-1615-4779-b8d6-69cd18f05ddf.png)







The system, roughly, consists of the following  stages, using a several algorithms in each stage:

The trajectory's initial estimation:
Since the Bundle Adjustment algorithm- which boils down to minimizing the projection error between the image locations of observed image points using nonlinear least-squares algorithm, the system is required for an initial estimation which supplied by the following approach:

The system match fours of points, between each sequence of two frames (two stereo pictures).
For each four points, the system triangulates two of the points in the first frame, which yields a 3D point in the "real world" and a matching point in the picture plane.
At this phase, the system created a points cloud, and using the PNP procedure (in order to get the matrix [R |t] ) the RANSAC algorithm emits the best [R |t] matrix .

E.g.:
![3](https://user-images.githubusercontent.com/82065601/196912422-a54bf1f3-f911-4980-9b42-ee722c7d26d6.png)
![4](https://user-images.githubusercontent.com/82065601/196912515-5c3bbdf8-c1af-4ef9-bb0a-6424ec7601df.png)
 ![5](https://user-images.githubusercontent.com/82065601/196912540-068448e3-d61f-495a-af46-d1369be8e7bb.png)


In the first phase the system performs Features matching using AKAZE algorithm with NORM_HAMMING (Those algorithm and norm were chosen after a trial-and-error (mostly ill-posed error) process and consultation with the experience of other fellow course members) .


In order to reject outliers, the system using Triangulation, Rectification and Significance tests, Blurring, and Consensus match.
While blurring is a quite common operation, the rectification test using the facts that the system is equipped with a stereo camera.
With regards to the Triangulation, the system rejects 3D points with negative z-value or "quite far" points.


Using those algorithms to reject outliers, the system accumulated the sequences' transformations between each two frames – using the PNP algorithm and the following lemma:

* Given the location of the camera as v = [x, y, z]T , [R| t] [x, y, z]T will yield the zero vector (since the camera position in the camera world is considered to be   the  system origin).
  Meaning, 	
  R v +t = 0   ->
  R v = -t  ->
  R-1Rv = R-1 -t  ->
  v = R-1-t   -> (since R is orthonormal)
  v = RT-t  = -RT t.

In order to implement the PNP algorithm the system accumulated the tracks along the trajectory and apply Consensus Match, RANSAC and Triangulation.
The initial estimation with respect to the ground truth:


































Accumulating relevant features Database.
As mentioned, in order to apply the Bundle Adjustment algorithm the system maintained a Database of the relevant data, which one can look at as a set of tracks along the trajectory (in this part), where each track is essentially a 3D point in the "real world" (landmark) that’s appears in a sequence of frames along the trajectory.

E.g. (marked with a small red dot):

 


Quantitative and qualitative analysis:

Total number of tracks:	152,389
Number of frames:	3,450
Max track length:	128
Min track length:	2
Mean track length:	4.972






















 














	
































Optimizing the estimation by Bundle Adjustment optimization:
At this point the system is starting to take into account the probability and the uncertainty aspects.
As mentioned above, since the Bundle Adjustment algorithm- which boils down to minimizing the projection error- thus, the minimization is achieved using nonlinear least-squares algorithms. Of these, Levenberg–Marquardt has proven to be one of the most successful due to its ease of implementation and the ability to converge quickly from a wide range of initial guesses.
Worth mentioning, when solving the minimization problems arising in the framework of bundle adjustment, the normal equations have a sparse block structure owing to the lack of interaction among parameters for different 3D points and cameras. This treat exploited to gain tremendous computational benefits by employing a sparse variant of the Levenberg–Marquardt algorithm which explicitly takes advantage of the normal equations zeros pattern, avoiding storing and operating on zero-elements.

Nevertheless, considering the complexity of the required computations, instead of adjusts one massive bundle (in our case for example 3450 frames), the system adjusts multiple local bundles with a small sequence of consecutive frames.
In order to adjusts each local bundle, the system create a factor graph based on the Database that was build and mentioned in the previous stage.

The bundle adjustment optimization as shown and submitted in stage 5:




























•	As can one indicate, the Bundle Adjustment optimization were did emits an optimization but not as quite as can be expected.
After extended research, and after observing that the system has a loose connectivity.
The previous part of the system massively changed, as describe previously:
o	The SIFT detector were replaced by AKAZE detector
o	L2 Norm replaced by HAMMING Norm.
o	Significance Test's ratio were reduced.
o	Extractor.
o	Thresholds detection for outliers.
o	RANSAC's iteration.
As can show those changes made the following connectivity improvement:


































Worth mentioning:  As can show by the project's process those changes were made only after the system suffered from an "ill posed" problems at the next stages – the Bundle Adjustment Algorithm.














































Performing Loop Closure After Creating a Pose Graph from the accumulated Data:
Moving along, to the last stage- the loop closure:
Essentially, loop closure is the task of deciding whether or not a vehicle has, after an excursion of arbitrary length, returned to a previously visited area.
This detection is of vital importance in the process of simultaneous localization and mapping as it helps to reduce the cumulative error of the estimated pose and generate a consistent "global map".
The graph that the system using in order to close a loop is the Factor Graph.
As mentioned above the bundle adjustment stage yield the "probability and the uncertainty aspects" by providing a relative poses and a covariance matrix.
(In this topic, the Mahalanobis distance is worth mentioning) 

Taking into account the complexity of this detection, the system spots a loop closure using the "funnel method":
Perform the heavy operations (e.g., Visual descriptor-based matching) only on suspects who passed the light operations (Geometric intersection).
After finding the shortest path using the Dijkstra algorithm, the relative keyframes'  covariance is the sum of the covariances who take part in the shortest path (as mentioned in the lecture, since most of the matrices' constrained covariance contains similar values, the edge's weights were all sets to 1.)

The heavy operation, that’s apply only on the suspects that passed the geometric intersection part, were implements as expected using the consensus match, such that any suspects which passed the chosen threshold (of inliers percentage and inliers amount) were added to the Pose Graph:
 

This routine applied for each keyframe (for each keyframe the system searched for a loop closure in the previous keyframes).

A graph of the absolute location error for the whole pose graph both with and without loop closures:










A graph of the location uncertainty size for the whole pose graph both with and without loop closures (log scaled):








Using the matrices' determinant as a measure for the uncertainty size (with and without loop closure).



















Discussion and Conclusions:
We can cautiously say that our final results are quite impressive, the system managed to map and to localize the car's environment, when equipped only with two stereo cameras(!).
Nevertheless, due to the importance of robustness and accuracy when dealing with such problems we can safely say that - there is still a lot of work to do.

A conspicuous weak spot in the system is the run time- those kinds of technologies are meant to run in real time and at this point it's still seemed to be  out of the system reach.
The system relies on several assumptions but according to the scoreboard those assumptions seem to be realistic (such as, The Gaussian distribution and the MLE assumption), those assumptions were crucial to improve the system's runtime and performance, and even make it feasibility (E.g., Adjusts multiple bundles rather than one "Incalculable" bundle).

Another conspicuous weak is dealing with the cumulative error. The system managed to reduce the cumulative error (which clearly seen as a massive error) only in cases were a loop closure were detected. Such "assumption"  is unrealistic, one cannot rely on that, and even if a loop closure is guaranteed the "price" for such error can be catastrophic (until one detected), a suggestion for that problem can be KLT.
Moreover, loop closure can help  retrospectively in mapping but cannot contribute in 'real time'.

In order to improve the system, one can suggest exploring and using features that are not inevitably in open-cv, for example super point, R2D2.
Another option, for the sake of connectivity improvement would be to check several matches for each point and examine which of them passes the detection of exceptions.



