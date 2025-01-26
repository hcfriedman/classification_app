import numpy as np
import cv2

def kmeans_clustering(image, k):

    assert k > 0, "k must be greater than 0"
    assert k < 100, "k must be less than 100"
    
    # Convert the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)

    # Define criteria and apply KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Create grayscale image where each class is a different intensity value
    # Reshape labels back to image dimensions and scale to 0-255 range
    labels = labels.reshape(image.shape[0], image.shape[1])
    labels = labels.astype(np.uint8)
    return labels

def get_display_classification(labels):
    # Display the classification with a color-blind friendly color palette
    cud_colors = [
        [0, 114, 178],    # Blue
        [86, 180, 233],   # Sky Blue
        [0, 158, 115],    # Green
        [240, 228, 66],   # Yellow
        [230, 159, 0],    # Orange
        [213, 94, 0],     # Red
        [204, 121, 167],  # Pink
        [153, 153, 153],  # Gray
    ]

    # Create a new image with the same dimensions as the original image
    display_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for i in range(np.unique(labels).shape[0]):
        display_image[labels == i] = cud_colors[i]
    return display_image

if __name__ == "__main__":
    image = cv2.imread(r"/Users/henryfriedman/Downloads/linkedin.png")
    NUM_CLUSTERS = 5
    labels = kmeans_clustering(image, NUM_CLUSTERS)
    cv2.imshow("KMeans Clustering", get_display_classification(labels))
    cv2.waitKey(0)
    cv2.destroyAllWindows()