import cv2
import numpy as np

def save_segmentation_masks(results, output_path):
    """
    Saves an image with only the segmentation masks drawn as black lines on a white background.
    """
    if not results:
        return

    h, w = results[0].orig_shape
    # Create a white canvas
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    if results[0].masks is not None:
        for segments in results[0].masks.xy:
            # Convert coordinates to integers
            points = np.array(segments, dtype=np.int32)
            
            # Draw path with black color (0, 0, 0)
            cv2.polylines(canvas, [points], isClosed=True, color=(0, 0, 0), thickness=2)

    cv2.imwrite(output_path, canvas)
    print(f"Saved segmentation masks (frame only) to '{output_path}'.")

def save_colored_masks(results, output_path):
    """
    Saves the original image with segmentation masks and bounding boxes overlaid.
    """
    if not results:
        return

    # Plot the results on the original image
    # plot() returns a BGR numpy array
    annotated_frame = results[0].plot()

    cv2.imwrite(output_path, annotated_frame)
    print(f"Saved colored masks (overlay) to '{output_path}'.")
