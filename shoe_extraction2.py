import numpy as np
import cv2
import matplotlib.pyplot as plt


# Load the shoe image in grayscale
image = cv2.imread('redshoes.jpg', cv2.IMREAD_GRAYSCALE)

if image is not None:
    # Compute the gradient magnitude
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
    
    
    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area or other criteria
    min_area_threshold = 5000  # Example threshold, adjust as needed
    shoe_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]

    # Create an empty mask
    mask = np.zeros_like(image)

    # Draw filled contours on the mask
    cv2.drawContours(mask, shoe_contours, -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Segment the original image globally using the computed histogram
    _, global_thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert the binary mask if needed
    if np.mean(global_thresholded_image) > 128:
        global_thresholded_image = 255 - global_thresholded_image
    
    # Apply the binary mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=global_thresholded_image)
    
    # Combine 'result' and 'masked_image' to produce a new masked image
    new_masked_image = cv2.bitwise_and(mask, masked_image)


    # Display the results
    plt.figure(figsize=(24, 12))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Binary Mask (Gradient Magnitude Thresholded)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(global_thresholded_image, cmap='gray')
    plt.title('Global Thresholded Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(new_masked_image, cmap='gray')
    plt.title('Masked Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
else:
    print("Image not found or unable to read!")


    
