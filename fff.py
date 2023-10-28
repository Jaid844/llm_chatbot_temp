import os

def generate_image(image_name, image_directory):
  """Generates an image, given the image name and image directory.

  Args:
    image_name: The name of the image.
    image_directory: The directory where the image is located.

  Returns:
    The image.
  """

  # Get the full path to the image file.
  image_path = os.path.join(image_directory, image_name)

  # Check if the image file exists.
  if not os.path.exists(image_path):
    raise FileNotFoundError(f"The image file {image_path} does not exist.")

  # Open the image file.
  with open(image_path, "rb") as image_file:
    image_bytes = image_file.read()

  return image_bytes

# Example usage:

image_name = "comics Batman HD Wallpaper.png"
image_directory = "downloads"

image_bytes = generate_image(image_name, image_directory)

# Display the image.
from PIL import Image
image = Image.open(image_bytes)
image.show()