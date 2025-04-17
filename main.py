import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import subprocess

# -----------------------------------------------------------------------------
# Function to get and print GPU memory statistics using nvidia-smi
# -----------------------------------------------------------------------------
def get_gpu_memory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE
    )
    gpu_memory = result.stdout.decode('utf-8').strip().split('\n')
    for i, memory in enumerate(gpu_memory):
        total, used, free = memory.split(',')
        print(f"GPU {i}: Total: {total} MB, Used: {used} MB, Free: {free} MB")

# Check GPU memory before model training
get_gpu_memory()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# -----------------------------------------------------------------------------
# Image Preprocessing and Tiling (Dimensions are determined automatically)
# -----------------------------------------------------------------------------
# Define the input folder containing the processed images
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_folder = os.path.join(script_dir, "Data")

# Initialize list to store image tiles and list to keep track of the years
processed_image_tiles = []
years = []

# Define the tile size (each tile will be 128x128 pixels)
tile_size = 128

# Loop over selected years (from 2005 to 2020 in 5-year intervals)
for year in range(2005, 2021, 5):
    processed_image_path = os.path.join(processed_folder, f'binary_GHS_{year}.png')

    # Load the processed image using OpenCV
    processed_image = cv2.imread(processed_image_path)
    if processed_image is None:
        print(f"Error loading image for year {year}")
        continue

    # Convert the image to grayscale
    processed_image_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    # Get actual image dimensions and calculate the maximum dimensions that allow complete tiles
    height, width = processed_image_gray.shape
    max_row = (height // tile_size) * tile_size  # Maximum row index for complete tiles
    max_col = (width // tile_size) * tile_size     # Maximum column index for complete tiles
    
    # Tiling the image into 128x128 pieces using computed dimensions
    for i in range(0, max_row, tile_size):
        for j in range(0, max_col, tile_size):
            tile = processed_image_gray[i:i + tile_size, j:j + tile_size]
            # The tile will always be 128x128 due to our calculation of max_row and max_col
            processed_image_tiles.append(tile)
    
    # Append the year to track the order
    years.append(year)

# -----------------------------------------------------------------------------
# Data Preparation: Reshape and Create Sequences of Tiles
# -----------------------------------------------------------------------------
# Convert the list of tiles to a NumPy array with data type uint8
all_samples = np.array(processed_image_tiles, dtype=np.uint8)

# Compute the number of tiles per image automatically using the first image's dimensions
# (Assuming all images have the same dimensions)
first_image = cv2.imread(os.path.join(processed_folder, f'binary_GHS_{years[0]}.png'))
first_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
height, width = first_gray.shape
max_row = (height // tile_size) * tile_size
max_col = (width // tile_size) * tile_size
tiles_per_image = (max_row // tile_size) * (max_col // tile_size)
print("Number of tiles per image is: ", tiles_per_image)

# Define the sequence length (number of consecutive years to consider)
sequence_length = 3

# Calculate the number of training samples based on available years and tiles per image
num_samples = (len(years) - sequence_length) * tiles_per_image

# Initialize empty arrays for input sequences (X_data) and target sequences (y_data)
# The shape: (number of samples, sequence_length, tile_size, tile_size, 1)
X_data = np.empty((num_samples, sequence_length, tile_size, tile_size, 1))
y_data = np.empty((num_samples, sequence_length, tile_size, tile_size, 1))

sample_index = 0

# Reconstruct sequences by iterating over each possible starting year and each tile
for i in range(len(years) - sequence_length):
    for tile_idx in range(tiles_per_image):
        start_tile = i * tiles_per_image + tile_idx
        X_sequence = all_samples[start_tile : start_tile + sequence_length * tiles_per_image : tiles_per_image]
        y_sequence = all_samples[start_tile + tiles_per_image : start_tile + (sequence_length + 1) * tiles_per_image : tiles_per_image]
        # Expand dimensions to add channel information
        X_data[sample_index] = np.expand_dims(X_sequence, axis=-1)
        y_data[sample_index] = np.expand_dims(y_sequence, axis=-1)
        sample_index += 1


# Convert X_data and y_data to type uint8
X_data = X_data.astype(np.uint8)
y_data = y_data.astype(np.uint8)

# -----------------------------------------------------------------------------
# Splitting Data into Training and Validation Sets
# -----------------------------------------------------------------------------
validation_split = 0.1
num_validation_samples = int(validation_split * len(X_data))
X_val = X_data[:num_validation_samples]
y_val = y_data[:num_validation_samples]
X_train = X_data[num_validation_samples:]
y_train = y_data[num_validation_samples:]

# -----------------------------------------------------------------------------
# Define Model Checkpoint Callback
# -----------------------------------------------------------------------------
checkpoint_callback = ModelCheckpoint(
    filepath='models/Urban_model_{epoch:02d}.keras',
    save_freq=30,  # Save the model every 30 epochs
    save_weights_only=False,
    save_best_only=False
)


# -----------------------------------------------------------------------------
# Define the ConvLSTM Model Architecture
# -----------------------------------------------------------------------------
seq = Sequential()

# First ConvLSTM2D layer with Batch Normalization
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(sequence_length, tile_size, tile_size, 1),
                   strides=(1, 1),
                   padding='same', activation='tanh', return_sequences=True))
seq.add(BatchNormalization())

# Second ConvLSTM2D layer with Batch Normalization
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same', activation='tanh', return_sequences=True))
seq.add(BatchNormalization())

# Third ConvLSTM2D layer with Batch Normalization
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same', activation='tanh', return_sequences=True))
seq.add(BatchNormalization())

# Fourth ConvLSTM2D layer with Batch Normalization
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same', activation='tanh', return_sequences=True))
seq.add(BatchNormalization())

# Final Conv3D layer to output the prediction sequence
seq.add(Conv3D(filters=1, kernel_size=(11, 11, 11), activation='sigmoid', padding='same', data_format='channels_last'))

# -----------------------------------------------------------------------------
# Compile and Summarize the Model
# -----------------------------------------------------------------------------
optimizer = Adam(learning_rate=1e-3)
seq.compile(optimizer=optimizer, loss='binary_crossentropy')
print(seq.summary())

# -----------------------------------------------------------------------------
# Train the Model
# -----------------------------------------------------------------------------
history = seq.fit(X_train, y_train, batch_size=32, epochs=150, validation_data=(X_val, y_val), callbacks=[checkpoint_callback])

# -----------------------------------------------------------------------------
# Save the Trained Model
# -----------------------------------------------------------------------------
model_save_path = 'models/Urban_model.h5'
try:
    seq.save(model_save_path)
    print(f"Model saved successfully to {model_save_path}")
except Exception as e:
    print(f"Failed to save the model: {e}")

# -----------------------------------------------------------------------------
# Plot Training and Validation Loss
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
