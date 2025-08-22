import os
import json
import py7zr
import shutil
import hashlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
from collections import Counter, defaultdict


SURFACE_TYPES = [
    "ConeSurfaceType",
    "CylinderSurfaceType",
    "EllipticalConeSurfaceType",
    "EllipticalCylinderSurfaceType",
    "NurbsSurfaceType",
    "PlaneSurfaceType",
    "SphereSurfaceType",
    "TorusSurfaceType"
]

SURFACE_TYPE_MAP = {srf_type: i for i, srf_type in enumerate(SURFACE_TYPES)}


def get_surface_type_vector(srf_type_one, srf_type_two):
    vec = [0] * len(SURFACE_TYPES)
    for srf_type in [srf_type_one, srf_type_two]:
        idx = SURFACE_TYPE_MAP.get(srf_type)
        if idx is not None:
            vec[idx] = 1
    return vec

def extract_7z_files(source_dir: str, dest_dir: str):
    """Extract all .7z files in the source directory to the destination directory"""
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    print(f"Extracting files from {source_dir} to {dest_dir}...")
    
    for file in os.listdir(source_dir):
        if file.endswith('.7z'):
            print(f"Processing {file}...")
            archive_path = os.path.join(source_dir, file)
            folder_name = Path(file).stem
            extract_path = os.path.join(dest_dir, folder_name)
            os.makedirs(extract_path, exist_ok=True)
            with py7zr.SevenZipFile(archive_path, mode='r') as archive:
                archive.extractall(path=extract_path)
            print(f"Extracted {file} to {extract_path}")
            extracted_files = os.listdir(extract_path)
            print(f"Number of files in {extract_path}: {len(extracted_files)}")


def get_dataset_statistics(directory: str):
    """Compute minimum and maximum number of occurences, components, bodies, joints, holes from all the .json files in the dataset and provide a summary"""
    min_occurrences = float('inf')
    max_occurrences = 0
    min_components = float('inf')
    max_components = 0
    min_bodies = float('inf')
    max_bodies = 0
    min_joints = float('inf')
    max_joints = 0
    min_holes = float('inf')
    max_holes = 0
    joint_type_counter = Counter()
    surface_type_counter = Counter()
    total_joint_count = 0
    total_surface_count = 0
    assembly_count = 0
    assembly_part_counts = []
    assembly_face_counts = []

    for folder in tqdm(os.listdir(directory)):
        folder_path = os.path.join(directory, folder)
        if not os.path.isdir(folder_path):
            continue
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            # print(f"Processing folder: {subfolder_path}")
            json_path = os.path.join(subfolder_path, "assembly.json")
            if os.path.isfile(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)

                joints = data.get("joints") or {}
                occurrences = data.get("occurrences") or {}
                components = data.get("components") or {}
                bodies = data.get("bodies") or {}
                holes = data.get("holes") or []
                contacts = data.get("contacts") or []

                num_occurrences = len(occurrences)
                num_components = len(components)
                num_bodies = len(bodies)
                num_joints = len(joints)
                num_holes = len(holes)
                assembly_part_counts.append(num_bodies)

                min_occurrences = min(min_occurrences, num_occurrences)
                max_occurrences = max(max_occurrences, num_occurrences)
                min_components = min(min_components, num_components)
                max_components = max(max_components, num_components)
                min_bodies = min(min_bodies, num_bodies)
                max_bodies = max(max_bodies, num_bodies)
                min_joints = min(min_joints, num_joints)
                max_joints = max(max_joints, num_joints)
                min_holes = min(min_holes, num_holes)
                max_holes = max(max_holes, num_holes)

                # Count joint types
                for joint in joints.values():
                    joint_type = joint.get("joint_motion", {}).get("joint_type", "Unknown")
                    joint_type_counter[joint_type] += 1
                    total_joint_count += 1

                # Count surface types in contacts
                for contact in contacts:
                    for entity_key in ["entity_one", "entity_two"]:
                        entity = contact.get(entity_key, {})
                        surface_type = entity.get("surface_type")
                        if surface_type:
                            surface_type_counter[surface_type] += 1
                            total_surface_count += 1
                
                properties = data.get("properties", {})
                face_count = properties.get("face_count", None)
                print("face_count:", face_count)
                if face_count is not None:
                    assembly_face_counts.append(face_count)

                assembly_count += 1

    if assembly_count == 0:
        print("No assembly.json files found.")
        return

    print(f"Processed {assembly_count} assemblies.")
    print(f"Occurrences: min={min_occurrences}, max={max_occurrences}")
    print(f"Components:  min={min_components}, max={max_components}")
    print(f"Bodies:      min={min_bodies}, max={max_bodies}")
    print(f"Joints:      min={min_joints}, max={max_joints}")
    print(f"Holes:       min={min_holes}, max={max_holes}")

    print("\nRanked joint types:")
    for joint_type, count in joint_type_counter.most_common():
        freq = count / total_joint_count * 100 if total_joint_count else 0
        print(f"  {joint_type}: {count} ({freq:.2f}%)")

    print("\nRanked contact surface types:")
    for surface_type, count in surface_type_counter.most_common():
        freq = count / total_surface_count * 100 if total_surface_count else 0
        print(f"  {surface_type}: {count} ({freq:.2f}%)")
    
    # --- New: Assembly and part/face statistics ---
    print("\nAssembly statistics (number of parts per assembly):")
    print(f"  Total assemblies: {len(assembly_part_counts)}")
    print(f"  Min parts: {min(assembly_part_counts) if assembly_part_counts else 0}")
    print(f"  Max parts: {max(assembly_part_counts) if assembly_part_counts else 0}")
    print(f"  Avg parts: {np.mean(assembly_part_counts) if assembly_part_counts else 0:.2f}")

    print("\nPart (body) face statistics:")
    if assembly_face_counts:
        print("\nAssembly face count statistics (from properties):")
        print(f"  Min faces: {min(assembly_face_counts)}")
        print(f"  Max faces: {max(assembly_face_counts)}")
        print(f"  Avg faces: {np.mean(assembly_face_counts):.2f}")

def process_subdir(args):
    subdir, directory, filtered_directory, min_bodies = args
    subdir_path = os.path.join(directory, subdir)
    if not os.path.isdir(subdir_path):
        return (0, 0, 0)
    json_path = os.path.join(subdir_path, "assembly.json")
    if not os.path.isfile(json_path):
        return (0, 0, 0)
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        bodies = data.get("bodies")
        if bodies is None:
            root = data.get("root", {})
            bodies = root.get("bodies", {})
        if len(bodies) >= min_bodies:
            shutil.copytree(subdir_path, os.path.join(filtered_directory, subdir), dirs_exist_ok=True)
            return (1, len(bodies), 0)
        else:
            return (0, 0, 1)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return (0, 0, 0)

def filtering_round_one(directory: str, filtered_directory: str, min_bodies: int = 3, max_workers: int = 4):
    """Remove assemblies based on the number of bodies, in parallel."""
    os.makedirs(filtered_directory, exist_ok=True)
    subdirs = os.listdir(directory)
    keep_counter = 0
    remove_counter = 0
    total_parts = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_subdir, (subdir, directory, filtered_directory, min_bodies)) for subdir in subdirs]

        results = []
        with tqdm(total=len(subdirs), desc="Filtering assemblies") as pbar:
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    for keep, parts, remove in results:
        keep_counter += keep
        total_parts += parts
        remove_counter += remove

    print(f"Total assemblies kept: {keep_counter}, total bodies: {total_parts}")
    print(f"Total assemblies removed: {remove_counter}")


def deduplicate_assemblies(directory: str, min_bodies: int = 3):
    """
    Remove duplicate assemblies (by file content) and keep only those with more than min_bodies bodies.
    """
    parts_to_folders = defaultdict(list)

    for folder in tqdm(os.listdir(directory)):
        folder_path = os.path.join(directory, folder)
        if not os.path.isdir(folder_path):
            continue
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            json_path = os.path.join(subfolder_path, "assembly.json")
            if not os.path.isfile(json_path):
                continue
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                bodies = data.get("bodies") or {}
                # Use sorted tuple of body IDs as the key
                body_ids = tuple(sorted(bodies.keys()))
                parts_to_folders[body_ids].append(subfolder_path)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")

    # Print statistics
    duplicate_groups = {k: v for k, v in parts_to_folders.items() if len(v) > 1}
    print(f"Number of unique duplicate groups by parts: {len(duplicate_groups)}")
    total_duplicates = sum(len(v) - 1 for v in duplicate_groups.values())
    print(f"Total duplicate assemblies by parts (excluding the first in each group): {total_duplicates}")
    if duplicate_groups:
        print("\nSample duplicate groups (body_ids: [folders]):")
        for k, v in list(duplicate_groups.items())[:5]:
            print(f"{k}:")
            for p in v:
                print(f"  {p}")

def generate_assembly_contacts(main_data_dir: str, output_file: str, debug_mode: bool = False, surface_type_extraction: bool = False):
    all_rows = []
    success_log = []
    failure_log = []
    
    subdirs = os.listdir(main_data_dir)
    subdirs = [subdir for subdir in subdirs if subdir.lower() != "contacts"]

    if debug_mode:
        print("Debug mode is enabled. Processing a single assembly. '7780_6c885e81'.")
        subdirs = ["7780_6c885e81"]

    for subdir in tqdm(subdirs, desc="Processing top-level subdirs"):
        subdir_path = os.path.join(main_data_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        assembly_json_path = os.path.join(subdir_path, "assembly.json")
        if not os.path.isfile(assembly_json_path):
            failure_log.append({"assembly_id": subdir, "reason": "assembly.json not found"})
            continue
        try:
            with open(assembly_json_path, "r") as f:
                data = json.load(f)
            
            contacts_list = data.get("contacts", [])
            if not contacts_list:
                failure_log.append({"assembly_id": subdir, "reason": "Contacts are null or empty."})
                continue
                
            contact_counts = defaultdict(lambda: {"contact_label": 0, "num_contacts": 0})
            
            # print(f"--- Debugging Parquet Generation for {subdir} ---")
            # Process each contact
            for i, contact in enumerate(contacts_list):
                entity_one = contact.get("entity_one", {})
                entity_two = contact.get("entity_two", {})
                
                # Get body and occurrence IDs
                body_one = entity_one.get("body")
                body_two = entity_two.get("body")
                occ_one = entity_one.get("occurrence")
                occ_two = entity_two.get("occurrence")
                
                # Create full IDs in the format occurrence_uuid_body_uuid
                id_one = f"{occ_one}_{body_one}" if occ_one else body_one
                id_two = f"{occ_two}_{body_two}" if occ_two else body_two
                
                if body_one and body_two:
                    # Create link ID in the same format as assembly_graph.py
                    link_id = f"{id_one}>{id_two}"
                    # print(f"  - Contact {i+1}: Creating link_id='{link_id}'")
                    contact_counts[link_id]["num_contacts"] += 1
                    contact_counts[link_id]["contact_label"] = 1
                else:
                    print(f"  - Contact {i+1}: SKIPPED (missing body_one or body_two)")

            # Create rows for all contacts
            rows = []
            print(f"  - Total unique link_ids created: {len(contact_counts)}")
            for link_id, counts in contact_counts.items():
                row = {
                    "assembly_id": subdir,
                    "link_id": link_id,
                    "contact_label": counts["contact_label"],
                    "num_contacts": counts["num_contacts"]
                }
                rows.append(row)
            print("--- End Debugging ---")

            all_rows.extend(rows)
            success_log.append({"assembly_id": subdir, "status": "Success"})
        except Exception as e:
            failure_log.append({"assembly_id": subdir, "reason": str(e)})
            print(f"Error processing {assembly_json_path}: {e}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_parquet(output_file, index=False)
    print(f"All contact labels saved to {output_file}")

    log_dir = os.path.dirname(output_file)
    success_log_path = os.path.join(log_dir, "contacts_success_log.json")
    failure_log_path = os.path.join(log_dir, "contacts_failure_log.json")

    with open(success_log_path, "w") as f:
        json.dump(success_log, f, indent=4)
    with open(failure_log_path, "w") as f:
        json.dump(failure_log, f, indent=4)

    print(f"Success log saved to {success_log_path}")
    print(f"Failure log saved to {failure_log_path}")
    print(f"Total successful assemblies: {len(success_log)}")
    print(f"Total failed assemblies: {len(failure_log)}")

def extract_contact_labels(assembly_json_path: str):
    with open(assembly_json_path, "r") as f:
        data = json.load(f)
    
    contacts = data.get("contacts", [])
    if contacts is None:
        return defaultdict(set)
    
    body_contacts = defaultdict(set)
    for contact in contacts:
        entity_one = contact.get("entity_one", {})
        entity_two = contact.get("entity_two", {})
        
        body_one = entity_one.get("body")
        body_two = entity_two.get("body")
        
        occ_one = entity_one.get("occurrence")
        occ_two = entity_two.get("occurrence")
        
        id_one = f"{occ_one}_{body_one}" if occ_one else body_one
        id_two = f"{occ_two}_{body_two}" if occ_two else body_two
        
        if body_one and body_two:
            body_contacts[id_one].add(id_two)
            body_contacts[id_two].add(id_one)
    
    return body_contacts

def print_unique_surface_types(main_data_dir: str):
    unique_surface_types = set()
    subdirs = os.listdir(main_data_dir)
    subdirs = [subdir for subdir in subdirs if subdir.lower() != "contacts"]

    for subdir in tqdm(subdirs, desc="Scanning top-level subdirs"):
        subdir_path = os.path.join(main_data_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for inner_dir in os.listdir(subdir_path):
            inner_dir_path = os.path.join(subdir_path, inner_dir)
            if not os.path.isdir(inner_dir_path):
                continue
            assembly_json_path = os.path.join(inner_dir_path, "assembly.json")
            if not os.path.isfile(assembly_json_path):
                continue
            try:
                with open(assembly_json_path, "r") as f:
                    data = json.load(f)
                contacts = data.get("contacts", [])
                if not isinstance(contacts, list):
                    contacts = []
                for contact in contacts:
                    for entity_key in ["entity_one", "entity_two"]:
                        entity = contact.get(entity_key, {})
                        surface_type = entity.get("surface_type")
                        if surface_type:
                            unique_surface_types.add(surface_type)
            except Exception as e:
                print(f"Error reading {assembly_json_path}: {e}")

    print("\nUnique surface types found in contacts:")
    for st in sorted(unique_surface_types):
        print(f"  {st}")

def save_contact_labels_to_parquet(contacts, assembly_id, output_path, debug_mode=False):
    """
    Save contact labels to a Parquet file, aggregating multiple contacts between two parts.

    Parameters:
        contacts (dict): Dictionary of contact labels.
        assembly_id (str): Unique identifier for the assembly.
        output_path (str): Path to save the Parquet file.
    """
    contact_counts = defaultdict(lambda: {"contact_label": 1, "num_contacts": 0})

    # Aggregate contacts
    for part_1, neighbors in contacts.items():
        for part_2 in neighbors:
            key = tuple(sorted([part_1, part_2]))  # Ensure consistent ordering of parts
            contact_counts[key]["num_contacts"] += 1
    rows = [
        {
            "assembly_id": assembly_id,
            "part_1": key[0],
            "part_2": key[1],
            "contact_label": contact_counts[key]["contact_label"],
            "num_contacts": contact_counts[key]["num_contacts"],
        }
        for key in contact_counts
    ]

    df = pd.DataFrame(rows)
    if debug_mode:
        print("Contact labels DataFrame:")
        print(df.head())
    df.to_parquet(output_path, index=False)
    print(f"Contact labels saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract 7z files and compute dataset statistics.")
    parser.add_argument("--extract", action='store_true', help="Extract .7z files")
    parser.add_argument("--source", type=str, help="Source directory containing .7z files")
    parser.add_argument("--destination", type=str, help="Destination directory for extracted files")
    parser.add_argument("--filterdir", type=str, help="Destination directory for filtered files")
    parser.add_argument("--deduplicate", action='store_true', help="Deduplicate assemblies based on body IDs")
    parser.add_argument("--getcontacts", action='store_true', help="Extract contact labels from assembly.json files")
    parser.add_argument("--surftype", action='store_true', help="Extract contact labels from assembly.json files")
    parser.add_argument("--stats", action='store_true', help="Compute dataset statistics from extracted files")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode for detailed output")
    args = parser.parse_args()

    if args.extract:
        if not os.path.exists(args.source):
            print(f"Source directory {args.source} does not exist.")
            return
        if not os.path.exists(args.destination):
            os.makedirs(args.destination)
        print(f"Extracting files from {args.source} to {args.destination}...")
        extract_7z_files(args.source, args.destination)

    if args.deduplicate:
        if not os.path.exists(args.destination):
            print(f"Destination directory {args.destination} does not exist.")
            return
        # filtering_round_one(args.destination, args.filterdir, min_bodies=4)
        filtering_round_one(args.destination, args.filterdir, min_bodies=2)
        # print("Running deduplication...")
        # deduplicate_assemblies(args.destination, min_bodies=3)

    if args.stats:
        print("Computing dataset statistics...")
        get_dataset_statistics(args.destination)
    
    if args.debug:
        generate_assembly_contacts(args.destination, os.path.join(args.destination, "contacts/contacts.parquet"), debug_mode=args.debug)

        # print_unique_surface_types(args.destination)
        # print("########################## DEBUG MODE ##########################")
        # debug_file = os.path.join(args.destination, "assembly.json")
        # if os.path.exists(debug_file):
        #     with open(debug_file, "r") as f:
        #         debug_data = json.load(f)
        #     print("Debug data loaded:")
        # else:
        #     print(f"No debug file found at {debug_file}")
    if args.getcontacts and args.surftype:
        print("Extracting contact labels and surface types from assembly.json files...")
        generate_assembly_contacts(args.destination, os.path.join(args.destination, "contacts_SurfaceTypes/contacts.parquet"), debug_mode=args.debug, surface_type_extraction=args.surftype)
    elif args.getcontacts:
        print("Extracting contact labels from assembly.json files...")
        generate_assembly_contacts(args.destination, os.path.join(args.destination, "contacts/contacts.parquet"), debug_mode=args.debug)
      

if __name__ == "__main__":
    main()