from loadimages import ImageLoader

# Define the paths
image_folder1 = '/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_1'
image_folder2 = '/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_2'
metadata_path = 'HAM10000_metadata'

image_loader = ImageLoader(
 image_folder1, 
 image_folder2, 
 metadata_path
)

# loading images

try:
    images, image_labels, image_ids = image_loader.load_images()
    print("Successfully loaded", len(images), "images.")
except Exception as e:
    print("Error loading images:", str(e))


# counting rows/instances in metadata to see if it matches the number of images
try:
    numberrows = image_loader.count_rows()
    print("There are", numberrows, "rows in metadata")
    image_loader.print_rows(2)

except Exception as e:
    print("Error loading images:", str(e))










