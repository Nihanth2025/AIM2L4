import cv2
import numpy as np
import matplotlib.pyplot as plt
def display_image(title,image):
  """utility function to display an RGB image"""
  plt.figure(figsize=(10,10))
  if len(image.shape) == 2:
    plt.imshow(image, cmap='gray')
  else:
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  plt.title(title)
  plt.axis('off')
  plt.show()
  
def interactive_edge_detection(image_path):
  image=cv2.imread(image_path)
  if image is None:
    print("Error: Could not read image.")
    return
  gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  display_image("Grayscale Image",gray_image)
  print("select an option")
  print("1. Sobel Edge Detection")
  print("2. Canny Edge Detection")
  print("3. Laplacian Edge Detection")
  print("4. Gaussian Blur Edge Detection")
  print("5. Meadian Filtering")
  print("6. Exit")
  while True:
    choice=input("Please enter your choice:")
    if choice=="1":
      sobel_x=cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=5)
      sobel_y=cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=5)
      combined_sobel=cv2.bitwise_or(sobel_x.astype(np.uint8),sobel_y.astype(np.uint8)) 
      display_image(combined_sobel,"Sobel Edge Detection")
    elif choice=="2":
      print("Adjust threshold values for Canny Edge Detection")
      low_threshold=int(input("Enter low threshold value:"))
      high_threshold=int(input("Enter high threshold value:"))
      edges=cv2.Canny(gray_image,low_threshold,high_threshold)
      display_image(edges,"Canny Edge Detection")
    elif choice=="3":
      laplacian=cv2.Laplacian(gray_image,cv2.CV_64F)
      display_image(laplacian.astype(np.uint8),"Laplacian Edge Detection")
    elif choice=="4":
      print("Adjust kernel size for Gaussian Blur Edge Detection")
      kernel_size=int(input("Enter kernel size:"))
      blurred_image=cv2.GaussianBlur(gray_image,(kernel_size,kernel_size),0)
      display_image("Gaussian Blur Edge Detection",blurred_image)
    elif choice=="5":
      print("Adjust kernel size for Median Filtering")
      kernel_size=int(input("Enter kernel size:"))
      median_image=cv2.medianBlur(gray_image,kernel_size)
      display_image("Median Filtered Image",median_image)
    elif choice=="6":
      break
    else:
      print("Invalid choice. Please try again.")
interactive_edge_detection("landscape.jpg")