from labelling import MainClass

# Define the paths
image_folder1 = '/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_1'
image_folder2 = '/Users/ninazorawska/Desktop/project 22/HAM10000_images_part_2'
metadata_path = 'HAM10000_metadata'

# Instantiate and run the main class
main_class = MainClass(image_folder1, image_folder2, metadata_path)
X_train, X_test, y_train, y_test, ids_train, ids_test = main_class.run()

# Now you have:
# - X_train, X_test: Image data
# - y_train, y_test: Disease labels
# - ids_train, ids_test: Image IDs

