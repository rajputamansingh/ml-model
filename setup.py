# Create a samples directory and add a placeholder MRI image
import os
from PIL import Image
import numpy as np

def create_sample_image():
    """Create a sample brain MRI-like image"""
    # Make samples directory if it doesn't exist
    if not os.path.exists("samples"):
        os.makedirs("samples")
    
    # Create brain tumor sample
    brain_tumor_path = os.path.join("samples", "default_brain_mri.jpg")
    if not os.path.exists(brain_tumor_path):
        # Create a simple brain MRI-like image
        width, height = 512, 512
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a gray ellipse for the brain
        for y in range(height):
            for x in range(width):
                # Ellipse equation: (x-h)²/a² + (y-k)²/b² <= 1
                a, b = 200, 240
                h, k = width//2, height//2
                if ((x-h)**2/a**2 + (y-k)**2/b**2) <= 1:
                    # Base brain color (gray)
                    gray_val = 180 - int(40 * ((x-h)**2/a**2 + (y-k)**2/b**2))
                    img_array[y, x] = [gray_val, gray_val, gray_val]
        
        # Add a "tumor" (lighter area)
        tumor_x, tumor_y = width//2 - 50, height//2 - 30
        tumor_radius = 40
        for y in range(height):
            for x in range(width):
                if ((x-tumor_x)**2 + (y-tumor_y)**2) <= tumor_radius**2:
                    # Make tumor area brighter
                    brightness = 255 - int(100 * ((x-tumor_x)**2 + (y-tumor_y)**2) / tumor_radius**2)
                    img_array[y, x] = [brightness, brightness, brightness]
        
        # Create and save the image
        img = Image.fromarray(img_array)
        img.save(brain_tumor_path)
        print(f"Created sample image at {brain_tumor_path}")

if __name__ == "__main__":
    create_sample_image()
    print("Setup completed successfully!")
