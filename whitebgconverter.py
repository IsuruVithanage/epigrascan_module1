import cv2
from pathlib import Path


def invert_images():
    # Define input and output directories
    input_dir = Path("data/segmented_chars")
    output_dir = Path("processed_images")

    # Ensure the input directory exists before proceeding
    if not input_dir.exists():
        print(f"Error: The folder '{input_dir}' was not found.")
        return

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the image extensions we want to look for
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    count = 0

    # Iterate through all files in the raw_images folder
    for file_path in input_dir.iterdir():
        if file_path.suffix.lower() in valid_extensions:
            # Read the image
            img = cv2.imread(str(file_path))

            if img is not None:
                # Invert the image colors (Black -> White, White -> Black)
                inverted_img = cv2.bitwise_not(img)

                # Define the path for the new image and save it
                output_path = output_dir / file_path.name
                cv2.imwrite(str(output_path), inverted_img)

                print(f"Processed: {file_path.name}")
                count += 1
            else:
                print(f"Warning: Could not read {file_path.name}. It might be corrupted.")

    print(f"\nSuccess! {count} images have been inverted and saved to the '{output_dir}' folder.")


if __name__ == "__main__":
    invert_images()