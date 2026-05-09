import cv2
import os


def crush_background(img_path):
    # SAFETY CHECK
    if not os.path.exists(img_path):
        print(f"❌ ERROR: Could not find the file at '{img_path}'")
        return

    # Read the image and convert to grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. FORCE THE BACKGROUND TO BLACK
    gray[gray < 100] = 0

    # 2. BOOST THE TEXT VISIBILITY
    enhanced_text = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

    # Make sure the image is strictly black and white for contour detection
    _, binary = cv2.threshold(enhanced_text, 10, 255, cv2.THRESH_BINARY)

    # 3. THE SURGICAL STRIKE (Contour Area Filtering)
    # Find all the individual white shapes in the image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cleaned_text = binary.copy()

    # Look at every single white shape we found
    for contour in contours:
        area = cv2.contourArea(contour)

        # THE ASSASSIN: If the shape is smaller than 100 pixels...
        if area < 100:
            # Paint the entire shape solid black!
            cv2.drawContours(cleaned_text, [contour], -1, 0, -1)

    # Save the output
    cv2.imwrite("super_black_background.jpg", cleaned_text)
    print("✅ Done! Background crushed and surgical noise removal applied.")


# Point exactly to where the image lives
image_location = "data/raw_estampages/raw_img1.jpg"

crush_background(image_location)