import os
import javabridge
import bioformats
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def convert_cxd_to_tiff(file_path_cxd, reConversionLogic=False):
    # Extract base directory and file name
    base_dir = os.path.dirname(file_path_cxd)
    file_name = os.path.splitext(os.path.basename(file_path_cxd))[0]

    # Define green and red folder paths
    green_folder = os.path.join(base_dir, f"{file_name}_green")
    red_folder = os.path.join(base_dir, f"{file_name}_red")

    # Define full file paths for saving TIFFs
    green_tiff_file = os.path.join(green_folder, f"{file_name}_green_ome.tiff")
    red_tiff_file = os.path.join(red_folder, f"{file_name}_red_ome.tiff")

    # Check if the green and red folders exist
    if not os.path.exists(green_folder):
        os.makedirs(green_folder)
        print(f"Created folder: {green_folder}")
        
    if not os.path.exists(red_folder):
        os.makedirs(red_folder)
        print(f"Created folder: {red_folder}")
    
    # Check if the TIFF files already exist
    if os.path.exists(green_tiff_file) and os.path.exists(red_tiff_file) and not reConversionLogic:
        print("Both TIFF files already exist. Skipping conversion.")
        return

    # Start the Java VM
    javabridge.start_vm(class_path=bioformats.JARS)

    try:
        # Read the image using Bio-Formats
        reader = bioformats.ImageReader(file_path_cxd)

        # Get the number of channels and frames
        num_channels = reader.rdr.getSizeC()  # Get the number of channels
        num_frames = reader.rdr.getImageCount()  # Get total frame count

        # Ensure there are exactly 2 channels (green and red)
        if num_channels != 2:
            raise ValueError(f"Expected 2 channels, but found {num_channels}")

        # Initialize stacks for green and red channels
        green_stack = []
        red_stack = []

        for i in range(0, num_frames, num_channels):  # Step by number of channels
            green_frame = reader.read(c=0, z=0, t=i // num_channels)   # Read green channel
            red_frame = reader.read(c=1, z=0, t=i // num_channels)  # Read red channel

            # Rotate green frame 90 degrees CCW
            green_frame_rot = np.rot90(green_frame)

            # Append to respective stacks
            green_stack.append(green_frame_rot)
            red_stack.append(red_frame)

            print(f"Processed frame {i // 2 + 1} / {num_frames // 2}")

        # Convert lists to 3D numpy arrays (for writing as stacks)
        green_stack = np.stack(green_stack, axis=0)
        red_stack = np.stack(red_stack, axis=0)

        # Save the stacks using Tifffile with zlib compression, using tqdm for the progress bar
        print(f"Saving {green_tiff_file}...")
        with tiff.TiffWriter(green_tiff_file, bigtiff=True) as tif_writer:
            for i in tqdm(range(green_stack.shape[0]), desc="Writing Green Stack"):
                tif_writer.write(green_stack[i], compression='zlib')

        print(f"Saving {red_tiff_file}...")
        with tiff.TiffWriter(red_tiff_file, bigtiff=True) as tif_writer:
            for i in tqdm(range(red_stack.shape[0]), desc="Writing Red Stack"):
                tif_writer.write(red_stack[i], compression='zlib')

        print(f"Saved green and red channel TIFFs successfully.")
    
    finally:
        # Stop the Java VM
        javabridge.kill_vm()

# Example usage
#convert_cxd_to_tiff("F:/Imaging/m1237/m1237_100224_task_day4_img/m1237_100224_task_day4_img_1.cxd", reConversionLogic=True)

def process_all_cxd_in_directory(directory_path):
    """
    This function finds all .cxd files in the provided directory and processes them
    using the convert_cxd_to_tiff function.
    
    Parameters:
        directory_path (str): Path to the directory containing .cxd files.
    """
    # List all files in the directory
    all_files = os.listdir(directory_path)
    
    # Filter for .cxd files only
    cxd_files = [file for file in all_files if file.endswith('.cxd')]
    
    # Check if there are any .cxd files
    if not cxd_files:
        print(f"No .cxd files found in {directory_path}.")
        return

    # Process each .cxd file
    for cxd_file in cxd_files:
        full_file_path = os.path.join(directory_path, cxd_file)
        print(f"Processing file: {full_file_path}")
        convert_cxd_to_tiff(full_file_path)

def batch_process_all_cxd_in_directory(root_directory):
    """
    This function iterates through all subdirectories of the given root directory,
    and for each subdirectory, it runs process_all_cxd_in_directory to process .cxd files.
    
    Parameters:
        root_directory (str): The root directory containing multiple subdirectories.
    """
    # List all subdirectories in the root directory
    subdirectories = [os.path.join(root_directory, sub_dir) for sub_dir in os.listdir(root_directory) 
                      if os.path.isdir(os.path.join(root_directory, sub_dir))]
    
    # Iterate through each subdirectory and process .cxd files
    for subdirectory in subdirectories:
        print(f"Processing folder: {subdirectory}")
        process_all_cxd_in_directory(subdirectory)

# Example usage
#batch_process_all_cxd_in_directory("F:/Imaging/m1237")

def visualize_mean_image(stack, channel_name='Image'):
    """
    Visualizes the mean image across all frames in a 3D stack.
    
    Parameters:
    stack (numpy.ndarray): 3D NumPy array where the first dimension is the frame number.
    channel_name (str): The name of the channel to display in the plot title (default is 'Image').
    
    Returns:
    None: The function displays the mean image.
    """
    # Compute the mean image across all frames (axis 0 is the frame axis)
    mean_image = np.mean(stack, axis=0)

    # Plot the mean image
    plt.figure(figsize=(8, 8))
    plt.imshow(mean_image, cmap='gray')
    plt.title(f"Mean {channel_name} Across All Frames")
    plt.colorbar()
    plt.show()

# Example usage:
#visualize_mean_image(green_stack, channel_name='Green Channel')