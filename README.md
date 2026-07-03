Program developped by 8cheh to use at CUPET only.
By 8cheh and Qwen(my favorite AI).

##Stucture
  AngleAP/
  ├── AP_app.py            # Flask Web 
  ├── AP_Picbetter.py      
  ├── AP_Segmenter.py      
  ├── AP_YLCalculator.py   # Young-Laplace
  ├── Test/
  │   ├── TE_contact_angle.py  # test tool
  │   └── drop_image.jpg       # test pic
  └── Enhancer/
      └── freeenhancer.py      # developping

##Key methods and core features
This project employs an automated pipeline to perform the complete measurement process, from droplet images to contact angle determination:
1. Droplet Localization — Automatically locates the droplet's position and size within the image using the Hough circle detection algorithm.
2. Image Enhancement — Improves the contrast and clarity of the droplet outline using CLAHE (Contrast Limited Adaptive Histogram Equalization) combined with Gaussian sharpening.
3. Droplet Segmentation — Separates the droplet from the background using Otsu's adaptive thresholding and morphological operations, while intelligently detecting the solid-liquid interface baseline.
4. Contact Angle Calculation — Numerically fits the droplet profile using the Young-Laplace equation (solving the ODE via the Runge-Kutta method combined with least-squares optimization); calculates the contact angles for both the left and right sides and computes their average.

##Two Usage methods
1.APP in 127.0.0.1:5000 by app.py (By flask):all by itself, no need for more clicks
2.Test/TE_contact_angle.py: click the 2 points by yourself and get better results(Maybe?)

##Technology Stack
1.Image Processing: OpenCV (Hough Circle Transform, CLAHE, Otsu's thresholding, morphological operations)
2.Numerical Computing: SciPy (ODE initial value problem solver `solve_ivp`, non-linear least squares `least_squares`)
3.Web Framework: Flask
4.Visualization: Matplotlib
5.Enhancement Module (Reserved): `Enhancer/freeenhancer.py` — planned integration with external APIs or local GPU for advanced image enhancement
