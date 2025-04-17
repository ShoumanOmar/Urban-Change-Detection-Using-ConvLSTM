# --------------------------------------------------------------------------
# Prediction Script for ConvLSTM Model Using 128x128 Tiles
# --------------------------------------------------------------------------
import numpy as np
import cv2
from keras.models import load_model
import os

# -----------------------------------------------------------------------------
# Set Paths and Parameters
# -----------------------------------------------------------------------------
# Folder containing the processed binary images (PNG format)
processed_folder = '/data2/UrbanGrowth/GHS/PNG'
# Tile size (each tile will be 128x128 pixels)
tile_size = 128
# Full image size (original image is 2000x2000 pixels, but last 80 pixels are skipped)
image_size = 2000

# Initialize lists for storing image tiles and corresponding years
processed_image_tiles = []
years = []

# -----------------------------------------------------------------------------
# Load and Preprocess Images for Each Year
# -----------------------------------------------------------------------------
# Process images for years 2010, 2015, and 2020. This results in a predcition for the year 2025
# To predict further, we use the predicted 2025 image and change the loop to range(2015,2026, 5)
for year in range(2010, 2021, 5):
    # Construct the file path for the current year's image
    processed_image_path = os.path.join(processed_folder, f'binary_GHS_{year}.png')
    
    # Load the image using OpenCV
    processed_image = cv2.imread(processed_image_path)
    if processed_image is None:
        print(f"Error loading image for year {year}")
        continue

    # Convert the loaded image to grayscale
    processed_image_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    # Convert binary values: 255 becomes 1 (so that binary operations work correctly)
    processed_image_gray[processed_image_gray == 255] = 1

    # -------------------------------------------------------------------------
    # Tiling the Image:
    # We are skipping the last 80 pixels to ensure complete tiles of size 128x128. 
    # Note that Here last 80 pixels are black and the same in all images. So no information is lost. 
    # You need to be careful when you do such operation with your image

    # -------------------------------------------------------------------------
    for i in range(0, image_size - 80, tile_size):  # Iterate over rows
        for j in range(0, image_size - 80, tile_size):  # Iterate over columns
            tile = processed_image_gray[i:i + tile_size, j:j + tile_size]
            # Confirm that the tile has the expected dimensions (128x128)
            if tile.shape == (128, 128):
                processed_image_tiles.append(tile)
    # Keep track of the year order
    years.append(year)

# -----------------------------------------------------------------------------
# Convert List of Tiles to a NumPy Array and Report the Data
# -----------------------------------------------------------------------------
test_samples = np.array(processed_image_tiles)
print(np.unique(test_samples), "all_samples Unique Values Before dividing by 255")
print(test_samples.shape, "Test Samples Shape")

# -----------------------------------------------------------------------------
# Determine the Number of Tiles per Image
# -----------------------------------------------------------------------------
# Calculate number of complete tiles in one image, considering the skipped pixels
tiles_per_image = ((image_size - 80) // tile_size) * ((image_size - 80) // tile_size)
print("Number of tiles per image is: ", tiles_per_image)

# -----------------------------------------------------------------------------
# Create Sequences of Tiles for the Model
# -----------------------------------------------------------------------------
# Instead of using full images, the model predicts on sequences of tiles.
# Define the sequence length (number of consecutive years/steps)
sequence_length = 3
# Calculate the total number of sample sequences.
# (len(years) - sequence_length + 1) ensures we cover all valid sequences.
num_samples = (len(years) - sequence_length + 1) * tiles_per_image

# Initialize array for test input sequences:
# Shape: (num_samples, sequence_length, tile_size, tile_size, 1)
X_test = np.empty((num_samples, sequence_length, tile_size, tile_size, 1))

sample_index = 0

# Reconstruct sequences for each tile across the time sequence
for i in range(len(years) - sequence_length + 1):
    for tile_idx in range(tiles_per_image):
        # Calculate the starting index for the current tile sequence in the flattened array
        start_tile = i * tiles_per_image + tile_idx
        # Select tiles for the current sequence across consecutive years
        tile_sequence = test_samples[start_tile:start_tile + sequence_length * tiles_per_image:tiles_per_image]
        # Expand dims to add the channel dimension (needed for the model input)
        X_test[sample_index] = np.expand_dims(tile_sequence, axis=-1)
        sample_index += 1

print(X_test.shape, "X_test shape")

# -----------------------------------------------------------------------------
# Load the Pre-Trained ConvLSTM Model
# -----------------------------------------------------------------------------
# Specify the path to the saved model file
model_save_path = '/data2/UrbanGrowth/models/convlstm_keras_binary_GHS_PhD3.h5'
# Load the model from disk
model = load_model(model_save_path)

# -----------------------------------------------------------------------------
# Make Predictions on the Test Data
# -----------------------------------------------------------------------------
predictions = model.predict(X_test)
print(predictions.shape, "predictions shape")
print(np.unique(predictions), "Predictions Unique Values")

# Select the last time step of the prediction sequence (i.e., the most recent prediction)
predicted_image = predictions[:, -1, :, :, :]

# -----------------------------------------------------------------------------
# Define Functions to Save and Reconstruct the Predicted Images
# -----------------------------------------------------------------------------
def save_predicted_images_with_threshold(predictions, save_folder, threshold=0.5):
    """
    Applies a threshold to the model's probability predictions to convert them
    into binary images, then saves each tile to disk.
    
    Args:
        predictions (numpy.ndarray): The model's prediction for each tile.
        save_folder (str): Directory to save the binary tile images.
        threshold (float): Threshold value for converting probabilities to binary.
        
    Returns:
        list: List of binary image tiles.
    """
    # Ensure the save folder exists; create it if it does not.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    saved_tiles = []
    for i, prediction in enumerate(predictions):
        # Convert probabilities to binary: values >= threshold become 255, otherwise 0.
        binary_prediction = (prediction[:, :, 0] >= threshold).astype(np.uint8) * 255
        # Construct the file path for saving the tile
        prediction_path = os.path.join(save_folder, f"tile_{i}.png")
        # Save the binary tile image
        cv2.imwrite(prediction_path, binary_prediction)
        saved_tiles.append(binary_prediction)
    
    return saved_tiles

def recreate_image_from_tiles(tiles, tile_size, image_size):
    """
    Reconstructs a full image from individual tile images.
    
    Args:
        tiles (list): List of image tiles.
        tile_size (int): Size of each tile (assumed square).
        image_size (int): Original full image size (will subtract skipped pixels).
        
    Returns:
        numpy.ndarray: The reconstructed full image.
    """
    # Adjust the full image size to account for skipped pixels (last 80 pixels are not used)
    adjusted_image_size = image_size - 80
    # Initialize an empty image array
    reconstructed_image = np.zeros((adjusted_image_size, adjusted_image_size), dtype=np.uint8)
    
    tile_index = 0
    # Iterate over the adjusted image dimensions to place each tile in order
    for i in range(0, adjusted_image_size, tile_size):
        for j in range(0, adjusted_image_size, tile_size):
            reconstructed_image[i:i + tile_size, j:j + tile_size] = tiles[tile_index]
            tile_index += 1
    
    return reconstructed_image

# -----------------------------------------------------------------------------
# Save Predicted Tiles and Reconstruct the Full Image
# -----------------------------------------------------------------------------
# Folder where predicted tiles and the reconstructed image will be saved
save_folder = '/data2/UrbanGrowth/GHS/Predictions'
# Save predicted tile images after thresholding (binary conversion)
saved_tiles = save_predicted_images_with_threshold(predicted_image, save_folder, threshold=0.5)

# Reconstruct the full image from the saved tiles
reconstructed_image = recreate_image_from_tiles(saved_tiles, tile_size=128, image_size=image_size)

# Save the final reconstructed image to disk
reconstructed_image_path = os.path.join(save_folder, "reconstructed_image.png")
cv2.imwrite(reconstructed_image_path, reconstructed_image)

print(f"Reconstructed image saved at {reconstructed_image_path}")
