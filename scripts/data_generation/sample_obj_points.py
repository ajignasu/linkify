import os
import json
import time
import random
import trimesh
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED


def save_points_as_ply(points: np.ndarray, filename: Path):
    """Save Nx3 points array to a PLY file in ASCII format."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")


class ParallelExecutor:
    """Lightweight wrapper to run tasks in parallel with progress bar and timeout handling."""

    def __init__(
        self,
        func,
        timeout: int | None = None,
        max_workers: int | None = None,
        log_interval: int = 300,
        desc: str = "Processing",
    ):
        self.func = func
        self.timeout = timeout
        self.log_interval = log_interval
        self.desc = desc

        # Determine sensible default for workers
        if max_workers is None:
            try:
                cores = multiprocessing.cpu_count()
            except NotImplementedError:
                cores = os.cpu_count() or 1
            # Use up to 8 workers, at least 1, minus one to keep a core free
            self.max_workers = max(1, min(8, cores - 1))
        else:
            self.max_workers = max_workers

        print(f"[ParallelExecutor] Using {self.max_workers} worker processes.")

    def run(
        self,
        items: list,
        success_log_path: Path | None = None,
        error_log_path: Path | None = None,
    ) -> Tuple[List[Tuple[Path, str]], dict]:
        """Execute self.func on each item in *items* using a process pool.

        Returns a tuple: (list_of_results, dict_of_errors)
        """
        results: List[Tuple[Path, str]] = []
        errors: dict = {}

        success_log = success_log_path.open("a", encoding="utf-8") if success_log_path else None
        error_log = error_log_path.open("a", encoding="utf-8") if error_log_path else None

        processed_counter = 0
        logged_timeouts = set()

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.func, item): item for item in items}
            start_times = {fut: time.time() for fut in futures}
            pending = set(futures.keys())

            pbar = tqdm(total=len(futures), desc=self.desc)

            while pending:
                done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)

                # process newly completed
                for future in done:
                    pending.remove(future)
                    item = futures[future]
                    try:
                        result = future.result()
                        results.append((item, result))
                        if success_log:
                            success_log.write(f"{item} | {result}\n")
                    except Exception as e:
                        errors[item] = str(e)
                        if error_log:
                            error_log.write(f"{item} | {e}\n")
                    processed_counter += 1
                    if processed_counter % self.log_interval == 0:
                        if success_log:
                            success_log.flush()
                        if error_log:
                            error_log.flush()
                    pbar.update(1)

                # timeout handling
                now = time.time()
                for future in list(pending):
                    if future in logged_timeouts:
                        continue
                    elapsed = now - start_times[future]
                    if self.timeout is not None and elapsed > self.timeout:
                        item = futures[future]
                        errors[item] = f"Timeout after {int(self.timeout)} seconds (still running)"
                        if error_log:
                            error_log.write(f"{item} | Timeout after {int(self.timeout)} seconds\n")
                        logged_timeouts.add(future)
                        processed_counter += 1
                        if processed_counter % self.log_interval == 0:
                            if success_log:
                                success_log.flush()
                            if error_log:
                                error_log.flush()

            pbar.close()

        if success_log:
            success_log.close()
        if error_log:
            error_log.close()

        return results, errors

def sample_points_from_obj(obj_path: Path, n_points: int) -> np.ndarray:
    """Load *obj_path* into trimesh and sample *n_points* points from its surface."""
    mesh = trimesh.load_mesh(obj_path, process=False)
    if mesh.is_empty or not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh from {obj_path}")

    # sample_surface_even can oversample; pick first n_points
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    if points.shape[0] != n_points:
        idx = np.random.choice(points.shape[0], n_points, replace=points.shape[0] < n_points)
        points = points[idx]
    return points.astype(np.float32)


def process_obj_file(
    args_tuple: Tuple[Path, Path, Path, int, bool]
) -> str:
    """Wrapper for multiprocessing. Samples points from a single OBJ file.
    Maintains the relative directory structure from source_root to output_root.
    """
    obj_path, source_root, output_root, n_points, force = args_tuple
    try:
        relative_path = obj_path.relative_to(source_root)
        ply_path = output_root / relative_path.with_suffix(".ply")
        if ply_path.exists() and not force:
            return f"skipped (exists)"
        points = sample_points_from_obj(obj_path, n_points)
        save_points_as_ply(points, ply_path)
        return f"processed"
    except Exception as e:
        raise RuntimeError(f"Error processing {obj_path.name}: {e}")


def parse_timeout_log_for_paths(log_path: Path) -> List[Path]:
    """Parses an error log to find file paths from lines containing 'Timeout'."""
    obj_paths_to_retry = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if "Timeout" in line:
                try:
                    start_index = line.find("'") + 1
                    end_index = line.find("'", start_index)
                    if start_index > 0 and end_index > start_index:
                        path_str = line[start_index:end_index]
                        obj_paths_to_retry.append(Path(path_str))
                except Exception as e:
                    print(f"[WARNING] Could not parse line: {line.strip()}. Error: {e}")
    return obj_paths_to_retry

def main():
    parser = argparse.ArgumentParser(description="Sample points on unique OBJ parts and export to PLY.")
    parser.add_argument("--source", type=str, help="Root directory containing OBJ files to process. Required for batch/retry modes.")
    parser.add_argument("--output", type=str, required=True, help="Output root directory for sampled PLY files.")
    parser.add_argument("--num", type=int, default=None, help="Max number of files to process (default: all)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: cores-1)")
    parser.add_argument("--points", type=int, default=2048, help="Number of points to sample per part")
    parser.add_argument("--logs", type=str, default=None, help="Directory to write success/error logs")
    parser.add_argument("--log-interval", type=int, default=300, help="Flush logs every N files")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per file in seconds")
    parser.add_argument("--force", action="store_true", help="Overwrite existing PLY files")
    parser.add_argument("--single-assembly", type=str, default=None, help="Debug mode. Provide a path to a single assembly directory to process.")
    parser.add_argument("--retry-from-log", type=str, default=None, help="Retry timed-out files from a given error log file.")
    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.single_assembly:
        assembly_dir = Path(args.single_assembly)
        if not assembly_dir.is_dir():
            print(f"[ERROR] Single assembly directory not found: {assembly_dir}")
            return
        print(f"[DEBUG] Processing single assembly directory: {assembly_dir}")
        if args.source:
            source_root = Path(args.source)
        else:
            source_root = assembly_dir.parent
        print(f"[DEBUG] Using source root for relative paths: {source_root}")
        obj_files = [
            p for p in assembly_dir.glob("*.obj")
            if p.is_file() and p.name.lower() != "assembly.obj"
        ]

        if not obj_files:
            print(f"No processable .obj files found in {assembly_dir}")
            return        
        success_log, error_log = None, None
        if args.logs:
            logs_dir = Path(args.logs)
            logs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            success_log_path = logs_dir / f"sample_debug_success_{timestamp}.log"
            error_log_path = logs_dir / f"sample_debug_errors_{timestamp}.log"
            success_log = success_log_path.open("a", encoding="utf-8")
            error_log = error_log_path.open("a", encoding="utf-8")
            print(f"[DEBUG] Logging success to {success_log_path}")
            print(f"[DEBUG] Logging errors to {error_log_path}")

        print(f"Found {len(obj_files)} files to process sequentially.")
        success_count = 0
        error_count = 0
        skipped_count = 0
        for obj_path in tqdm(obj_files, desc="Debug processing"):
            try:
                summary = process_obj_file(
                    (obj_path, source_root, output_dir, args.points, args.force)
                )
                if "skipped" in summary:
                    skipped_count += 1
                else:
                    success_count += 1
                if success_log:
                    success_log.write(f"{obj_path} | {summary}\n")
            except Exception as e:
                print(f"❌ Error processing {obj_path.name}: {e}")
                error_count += 1
                if error_log:
                    error_log.write(f"{obj_path} | {e}\n")
        
        if success_log: success_log.close()
        if error_log: error_log.close()

        print("\n=== DEBUG SUMMARY ===")
        print(f"✅ Processed successfully: {success_count}")
        print(f"⏩ Skipped (already exist): {skipped_count}")
        print(f"❌ Files with errors: {error_count}")
        print(f"Total files found: {len(obj_files)}")
        return

    if not args.source:
        print("[ERROR] --source is required for this mode.")
        return

    source_dir = Path(args.source)
    obj_files_to_process = []

    if args.retry_from_log:
        log_path = Path(args.retry_from_log)
        print(f"[INFO] Retrying timed-out files from log: {log_path}")
        obj_files_to_process = parse_timeout_log_for_paths(log_path)
    else:  # Default batch mode
        if not source_dir.is_dir():
            print(f"[ERROR] Source directory not found: {source_dir}")
            return

        # Find all .obj files recursively, excluding 'assembly.obj'
        print(f"Searching for .obj files in {source_dir}...")
        candidate_files = [
            p for p in source_dir.glob("**/*.obj")
            if p.is_file() and p.name.lower() != "assembly.obj"
        ]
        print(f"Found {len(candidate_files)} candidate .obj files.")
        if not args.force:
            print("Checking for existing files to skip...")
            for path in tqdm(candidate_files, desc="Checking existing files"):
                relative_path = path.relative_to(source_dir)
                ply_path = output_dir / relative_path.with_suffix(".ply")
                if not ply_path.exists():
                    obj_files_to_process.append(path)
            print(f"Found {len(obj_files_to_process)} new files to process.")
        else:
            obj_files_to_process = candidate_files
            print("Force flag is set, will process all candidate files.")


    if args.num is not None:
        obj_files_to_process = obj_files_to_process[: args.num]

    if not obj_files_to_process:
        print("No files to process. Exiting.")
        return

    print(f"Processing {len(obj_files_to_process)} files.")

    # logging
    logs_dir = Path(args.logs) if args.logs else output_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_prefix = "sample_retry" if args.retry_from_log else "sample"
    success_log = logs_dir / f"{log_prefix}_success_{timestamp}.log"
    error_log = logs_dir / f"{log_prefix}_errors_{timestamp}.log"

    items_to_process = [
        (obj_path, source_dir, output_dir, args.points, args.force)
        for obj_path in obj_files_to_process
    ]

    executor = ParallelExecutor(
        process_obj_file,
        timeout=args.timeout,
        max_workers=args.workers,
        log_interval=args.log_interval,
        desc="Sampling parts",
    )

    results, errors = executor.run(items_to_process, success_log, error_log)
    print("\n=== SUMMARY ===")
    print(f"✅ Successful files: {len(results)}")
    print(f"❌ Files with errors: {len(errors)}")

if __name__ == "__main__":
    main()