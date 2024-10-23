import os
import javabridge
import bioformats
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import re

class CxdBatchProcessor:
    def __init__(self, root_directory, skipIrrelevantFrames=True, reConversionLogic=False):
        """
        Initialize the batch processor with the root directory.
        
        Parameters:
        root_directory (str): The root directory containing multiple subdirectories.
        skipIrrelevantFrames (bool): If True, skips irrelevant frames as described. If False, takes all frames.
        reConversionLogic (bool): If True, forces reconversion even if TIFFs already exist.
        """
        self.root_directory = root_directory
        self.skipIrrelevantFrames = skipIrrelevantFrames  # Store as instance attribute
        self.reConversionLogic = reConversionLogic  # Store as instance attribute

    def start_jvm(self):
        """Start the JVM."""
        try:
            javabridge.kill_vm()  # Ensure any previous JVM is stopped
        except:
            pass  # Ignore error if no VM was running

        javabridge.start_vm(class_path=bioformats.JARS)  # Start JVM once

    def stop_jvm(self):
        """Stop the JVM."""
        javabridge.kill_vm()  # Kill JVM when processing is done

    def convert_cxd_to_tiff(self, file_path_cxd):
        """
        Convert a .cxd file to TIFF format, splitting channels into green and red.
        If skipIrrelevantFrames is True, the green stack will take frames 0, 2, 8, 10... 
        and the red stack will take frames 5, 7, 13, 15...
        If skipIrrelevantFrames is False, the function will take all frames for both channels.
        
        Parameters:
        file_path_cxd (str): Path to the .cxd file to convert.
        """
        base_dir = os.path.dirname(file_path_cxd)
        file_name = os.path.splitext(os.path.basename(file_path_cxd))[0]
    
        green_folder = os.path.join(base_dir, f"{file_name}_green")
        red_folder = os.path.join(base_dir, f"{file_name}_red")
    
        green_tiff_file = os.path.join(green_folder, f"{file_name}_green_ome.tiff")
        red_tiff_file = os.path.join(red_folder, f"{file_name}_red_ome.tiff")
    
        # Check if the green and red folders exist, create if necessary
        os.makedirs(green_folder, exist_ok=True)
        os.makedirs(red_folder, exist_ok=True)
    
        # Skip if both files already exist and reConversionLogic is False
        if os.path.exists(green_tiff_file) and os.path.exists(red_tiff_file) and not self.reConversionLogic:
            print(f"Both TIFF files already exist. Skipping conversion of {file_path_cxd}.")
            return
    
        # Read the .cxd file
        reader = bioformats.ImageReader(file_path_cxd)
        num_frames = reader.rdr.getImageCount()  # Total of 6800 frames (2 channels per logical frame)
    
        # Initialize stacks for green and red channels
        green_stack, red_stack = [], []
    
        if self.skipIrrelevantFrames:  # Use the class attribute here
            # Use tqdm for the progress bar
            for i in tqdm(range(0, num_frames, 8), desc="Processing frames", unit="frame"):
                # Green channel (take 0, 2, 8, 10...)
                if i < num_frames:
                    green_frame_1 = reader.read(c=0, z=0, t=i // 2)
                    green_stack.append(np.rot90(green_frame_1))
                if i+2 < num_frames:
                    green_frame_2 = reader.read(c=0, z=0, t=(i+2) // 2)
                    green_stack.append(np.rot90(green_frame_2))
    
                # Red channel (take 5, 7, 13, 15...)
                if i+5 < num_frames:
                    red_frame_1 = reader.read(c=1, z=0, t=(i+5) // 2)
                    red_stack.append(red_frame_1)
                if i+7 < num_frames:
                    red_frame_2 = reader.read(c=1, z=0, t=(i+7) // 2)
                    red_stack.append(red_frame_2)
        else:  # Take all frames without skipping
            # Process all frames for both channels
            for i in tqdm(range(0, num_frames, 2), desc="Processing all frames", unit="frame"):
                # Green channel (take all frames for channel 0)
                green_frame = reader.read(c=0, z=0, t=i // 2)
                green_stack.append(np.rot90(green_frame))
    
                # Red channel (take all frames for channel 1)
                red_frame = reader.read(c=1, z=0, t=i // 2)
                red_stack.append(red_frame)
    
        # Convert to 3D arrays
        green_stack = np.stack(green_stack, axis=0)
        red_stack = np.stack(red_stack, axis=0)
    
        # Save TIFF files with a progress bar
        self._save_tiff_with_progress(green_stack, green_tiff_file, "Writing Green Stack")
        self._save_tiff_with_progress(red_stack, red_tiff_file, "Writing Red Stack")
    
        print(f"Saved green and red channel TIFFs for {file_path_cxd} successfully.")

    def _save_tiff_with_progress(self, stack, file_path, description):
        """
        Save a 3D numpy array as a TIFF file with a progress bar.
        
        Parameters:
        stack (numpy.ndarray): The 3D stack to save as a TIFF.
        file_path (str): The path where the TIFF file will be saved.
        description (str): Description for the progress bar.
        """
        print(f"Saving {file_path}...")
        with tiff.TiffWriter(file_path, bigtiff=True) as tif_writer:
            for i in tqdm(range(stack.shape[0]), desc=description):
                tif_writer.write(stack[i], compression='zlib')

    def process_all_cxd_in_directory(self, directory_path):
        """
        Process all .cxd files in a given directory.

        Parameters:
        directory_path (str): Path to the directory containing .cxd files.
        """
        all_files = os.listdir(directory_path)
        cxd_files = [file for file in all_files if file.endswith('.cxd')]

        if not cxd_files:
            print(f"No .cxd files found in {directory_path}.")
            return

        for cxd_file in cxd_files:
            full_file_path = os.path.join(directory_path, cxd_file)
            print(f"Processing file: {full_file_path}")
            self.convert_cxd_to_tiff(full_file_path)

    def batch_process_all_cxd_in_directory(self, provided_directory=None):
        """
        Batch process all .cxd files in a provided directory or all subdirectories of the root directory.

        Parameters:
        provided_directory (str or list): Optional. A specific directory or list of directories to process instead of the root directory.
        """
        try:
            self.start_jvm()  # Start JVM once

            # Check if provided_directory is a list of directories
            if isinstance(provided_directory, list):
                for directory in provided_directory:
                    print(f"Processing provided directory: {directory}")
                    self.process_all_cxd_in_directory(directory)
            
            # If it's a single directory
            elif isinstance(provided_directory, str):
                print(f"Processing provided directory: {provided_directory}")
                self.process_all_cxd_in_directory(provided_directory)

            # If no directory is provided, process all subdirectories in the root
            else:
                subdirectories = [os.path.join(self.root_directory, sub_dir) for sub_dir in os.listdir(self.root_directory)
                                  if os.path.isdir(os.path.join(self.root_directory, sub_dir))]

                for subdirectory in subdirectories:
                    print(f"Processing folder: {subdirectory}")
                    self.process_all_cxd_in_directory(subdirectory)
        finally:
            self.stop_jvm()  # Ensure JVM is shut down at the end

    def visualize_mean_image(self, stack, channel_name='Image'):
        """
        Visualizes either a single 2D image or the mean image across all frames in a 3D stack.
    
        Parameters:
        stack (numpy.ndarray): Either a 2D array (single image) or a 3D array (image stack).
        channel_name (str): The name of the channel to display in the plot title (default is 'Image').
        """
        # Check if input is 2D (single image) or 3D (image stack)
        if stack.ndim == 2:
            # It's a 2D image
            image_to_visualize = stack
            title = f"{channel_name} (Single Image)"
        elif stack.ndim == 3:
            # It's a 3D stack, compute the mean image across all frames
            image_to_visualize = np.mean(stack, axis=0)
            title = f"Mean {channel_name} Across All Frames"
        else:
            raise ValueError("Input must be a 2D or 3D array.")
    
        # Visualize the image
        plt.figure(figsize=(8, 8))
        plt.imshow(image_to_visualize, cmap='gray')
        plt.title(title)
        plt.colorbar()
        plt.show()

    def batch_move_files_to_targetDir(self, source_directory=None, target_directory=None):
        """
        Copy the entire '_green' and '_red' folders containing converted ome.tiff files from the source directories to the target directory.
        
        Parameters:
        source_directory (str or list): The source directory or list of directories containing '_green' or '_red' subfolders.
        target_directory (str): The single target directory where the OME-TIFF files will be copied.
        """
        if target_directory is None:
            print("Target directory is not provided.")
            return

        # Validate target directory exists
        if not os.path.isdir(target_directory):
            print(f"Target directory does not exist: {target_directory}")
            return

        # Convert source_directory to list if it is a string or None
        if isinstance(source_directory, str):
            source_directories = [source_directory]
        elif isinstance(source_directory, list):
            source_directories = source_directory
        else:
            source_directories = [self.root_directory]  # Default to the root directory

        # Process each source directory
        for src_dir in source_directories:
            if not os.path.isdir(src_dir):
                print(f"Source directory does not exist: {src_dir}")
                continue

            # Recursively find subdirectories containing '_green' or '_red'
            for root, dirs, files in os.walk(src_dir):
                for dir_name in dirs:
                    if '_green' in dir_name or '_red' in dir_name:
                        # Extract the 6-digit date from the folder name using regex
                        match = re.search(r'\d{6}', dir_name)
                        if not match:
                            print(f"No 6-digit date found in folder name: {dir_name}")
                            continue

                        date_code = match.group(0)

                        # Look for the corresponding folder in the target directory with the 6-digit date
                        target_subdir = self.find_target_subdir_by_date(target_directory, date_code)
                        if not target_subdir:
                            print(f"No target subfolder found for date {date_code} in {target_directory}. Skipping.")
                            continue

                        # Find the subdirectory within the target directory that matches the full folder name
                        name_matching_dir = self._find_matching_subdir(target_subdir, os.path.basename(root))
                        if name_matching_dir is None:
                            print(f"No matching subdirectory found in target for {os.path.basename(root)}. Skipping.")
                            continue

                        # Get the full path of the folder containing the OME-TIFF files (either green or red)
                        source_folder_to_copy = os.path.join(root, dir_name)
                        print(f"Found matching subdirectory: {name_matching_dir}")

                        # Copy the entire '_green' or '_red' folder to the target folder
                        self._copy_folder_with_progress(source_folder_to_copy, name_matching_dir)

    def find_target_subdir_by_date(self, target_directory, date_code):
        """
        Search for a subdirectory that contains the given date_code.
        """
        target_subdir = None

        # Walk through the target_directory to find the folder with date_code
        for root, dirs, files in os.walk(target_directory):
            for dir_name in dirs:
                if date_code in dir_name:
                    target_subdir = os.path.join(root, dir_name)
                    break
            if target_subdir:
                break

        if target_subdir:
            print(f"Found target subdirectory: {target_subdir}")
        else:
            print(f"No matching subdirectory with date code {date_code} found in {target_directory}.")

        return target_subdir

    def _find_matching_subdir(self, target_subdir, dir_name):
        """
        Recursively search for a folder within target_subdir that matches dir_name.
        """
        for root, dirs, files in os.walk(target_subdir):
            if dir_name in dirs:
                return os.path.join(root, dir_name)
        return None

    def _copy_folder_with_progress(self, source_folder, target_folder):
        """
        Copy the entire folder from the source to the target folder with a progress bar.
        
        Parameters:
        source_folder (str): The path to the source folder containing OME-TIFF files.
        target_folder (str): The path to the target folder where files will be copied.
        """
        # Define the target path where the folder should be copied
        destination_folder = os.path.join(target_folder, os.path.basename(source_folder))

        print(f"Copying folder {source_folder} to {destination_folder}")

        try:
            # Use shutil.copytree() to copy the folder
            shutil.copytree(source_folder, destination_folder)

            print(f"Successfully copied folder {source_folder} to {destination_folder}")
        except Exception as e:
            print(f"Failed to copy folder {source_folder}: {e}")


'''
Example usage (file conversion):
    from cxd_processor import CxdBatchProcessor
    # Initialize the batch processor with the root directory
    processor = CxdBatchProcessor("F:/Imaging/m1237", skipIrrelevantFrames=True, reConversionLogic=True)
    # To run all cxd files in subdirectories
    processor.batch_process_all_cxd_in_directory()
    # To process a specific directory
    processor.batch_process_all_cxd_in_directory(provided_directory="F:/Imaging/m1237/subfolder")
    # To process a list of specific directories
    processor.batch_process_all_cxd_in_directory(provided_directory=["F:/Imaging/m1237/folder1", "F:/Imaging/m1237/folder2"])
Example usage (copy files to the target directory):
    processor = CxdBatchProcessor("F:/Imaging/m1237", skipIrrelevantFrames=True, reConversionLogic=True)
    processor.batch_move_files_to_targetDir(source_directory="F:/Imaging/m1237", target_directory="Z:/Rodent Data/dualImaging_parkj/m1237_GCAMP")
'''
