import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("upsample_contacts.log")
    ]
)
logger = logging.getLogger(__name__)

def upsample_point_cloud_with_replacement(pc, num_points=2048):
    """
    Upsample a point cloud to num_points using random sampling with replacement.
    Args:
        pc: np.ndarray of shape (N, 3) or (N, D)
        num_points: int, desired number of points
    Returns:
        np.ndarray of shape (num_points, 3) or (num_points, D)
    """
    N = pc.shape[0]
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
    else:
        idx = np.random.choice(N, num_points, replace=True)
    return pc[idx]

def should_process_ply_file(ply_path, target_points=2048):
    """
    Check if a .ply file has fewer than target_points by reading its header.
    Args:
        ply_path (str): Path to the .ply file.
        target_points (int): Threshold for the number of points.
    Returns:
        bool: True if the file has fewer than target_points, False otherwise.
    """
    try:
        with open(ply_path, 'r') as file:
            for i, line in enumerate(file):
                if i == 2 and line.startswith("element vertex"):
                    num_points = int(line.split()[-1])
                    return num_points < target_points
                if i > 4:
                    break
    except Exception as e:
        logger.error(f"Error reading header of {ply_path}: {e}")
    return False

def count_and_upsample_points_in_ply(ply_path, target_points=2048):
    """
    Count the number of points in a .ply file and upsample if necessary.
    """
    try:
        if not should_process_ply_file(ply_path, target_points):
            logger.info(f"Skipping {ply_path} as it has >= {target_points} points.")
            return 0

        ply_data = PlyData.read(ply_path)
        points = np.array([list(vertex) for vertex in ply_data['vertex'].data])
        num_points = len(points)
        if num_points < target_points:
            upsampled_points = upsample_point_cloud_with_replacement(points, num_points=target_points)
            vertex = np.array([tuple(point) for point in upsampled_points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            ply_element = PlyElement.describe(vertex, 'vertex')
            PlyData([ply_element], text=True).write(ply_path)
        logger.info(f"Successfully processed {ply_path} with {num_points} points.")
        return num_points
    except Exception as e:
        logger.error(f"Error processing {ply_path}: {e}")
        return 0

def analyze_ply_points_parallel(data_dir, max_workers=8, debug=False):
    """
    Analyze the number of points in .ply files across multiple assemblies in the given data directory.
    Upsample point clouds to 2048 points if they have fewer than 2048 points.

    Parameters:
        data_dir (str): Path to the root directory containing multiple assembly folders.
        max_workers (int): Number of threads for parallel processing.
        debug (bool): If True, process only one assembly for debugging.

    Returns:
        tuple: (total_ply_files, min_points, mean_points, max_points)
    """
    total_ply_files = 0
    point_counts = []
    ply_file_paths = []
    assembly_ids = os.listdir(data_dir)
    if debug:
        assembly_ids = assembly_ids[:1]

    for assembly_id in assembly_ids:
        assembly_path = os.path.join(data_dir, assembly_id)
        if os.path.isdir(assembly_path):
            contact_path = os.path.join(assembly_path, "contact")
            if os.path.exists(contact_path):
                ply_files = [os.path.join(contact_path, f) for f in os.listdir(contact_path) if f.endswith('.ply')]
                ply_file_paths.extend(ply_files)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for points in tqdm(executor.map(count_and_upsample_points_in_ply, ply_file_paths), desc="Processing .ply files", total=len(ply_file_paths)):
            if points > 0:
                point_counts.append(points)
            total_ply_files += 1

    # stats
    if point_counts:
        min_points = min(point_counts)
        mean_points = sum(point_counts) / len(point_counts)
        max_points = max(point_counts)
    else:
        min_points = 0
        mean_points = 0
        max_points = 0

    logger.info(f"Total number of .ply files: {total_ply_files}")
    logger.info(f"Minimum points in a .ply file: {min_points}")
    logger.info(f"Mean points in .ply files: {mean_points:.2f}")
    logger.info(f"Maximum points in a .ply file: {max_points}")

    return total_ply_files, min_points, mean_points, max_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and upsample .ply files.")
    parser.add_argument("--data_dir", type=str, help="Path to the root directory containing assembly folders.")
    parser.add_argument("--max_workers", type=int, default=32, help="Number of threads for parallel processing.")
    parser.add_argument("--debug", action="store_true", help="Run the script in debug mode for one assembly.")
    args = parser.parse_args()

    total_ply_files, min_points, mean_points, max_points = analyze_ply_points_parallel(
        args.data_dir, max_workers=args.max_workers, debug=args.debug
    )
    print(f"Total number of .ply files: {total_ply_files}")
    print(f"Minimum points in a .ply file: {min_points}")
    print(f"Mean points in .ply files: {mean_points:.2f}")
    print(f"Maximum points in a .ply file: {max_points}")