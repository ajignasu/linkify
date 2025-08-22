import os
import json
import random
import time
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from datetime import datetime
from scripts.data_generation.assemblyGraphGeneration.assembly_graph import AssemblyGraph
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError, wait, FIRST_COMPLETED

from OCC.Core.gp import gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Extend.DataExchange import read_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_SHELL
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.GProp import GProp_GProps
from OCC.Display.SimpleGui import init_display


class ParallelExecutor:
    def __init__(self, func, timeout=None, max_workers=None, log_interval=100, desc="Processing"):
        """
        func        : function to run on each item
        timeout     : max seconds to wait per task (None for no timeout)
        max_workers : number of worker processes
        desc        : progress bar description
        """
        self.func = func
        self.timeout = timeout
        self.log_interval = log_interval
        if max_workers is None:
            try:
                cores = multiprocessing.cpu_count()
            except NotImplementedError:
                cores = os.cpu_count() or 1
            self.max_workers = max(1, min(4, cores - 1))
            print(f"[ParallelExecutor] Detected {cores} CPU cores ➜ using {self.max_workers} worker processes.")
        else:
            self.max_workers = max_workers
            print(f"[ParallelExecutor] Using user-specified {self.max_workers} worker processes.")
        self.desc = desc

    def run(self, items, success_log_path: Path | None = None, error_log_path: Path | None = None):
        results = []
        errors = {}
        success_log = open(success_log_path, "a", encoding="utf-8") if success_log_path else None
        error_log = open(error_log_path, "a", encoding="utf-8") if error_log_path else None

        processed_counter = 0
        logged_timeouts = set()

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.func, item): item for item in items}
            start_times = {fut: time.time() for fut in futures}
            pending = set(futures.keys())

            pbar = tqdm(total=len(futures), desc=self.desc)

            while pending:
                done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.remove(future)
                    item = futures[future]
                    try:
                        result = future.result()
                        if ("Number of generated contacts: 0" in result or 
                            "ERROR:" in result or 
                            "no contacts" in result):
                            
                            errors[item] = result.split(" | ")[-1]  # Extract just the error part
                            if error_log:
                                error_log.write(f"{item} | {result.split(' | ')[-1]}\n")
                        else:
                            results.append((item, result))
                            if success_log:
                                success_log.write(f"{item} | {result}\n")
                    except Exception as e:
                        errors[item] = str(e)
                        if error_log:
                            error_log.write(f"{item} | {str(e)}\n")
                    processed_counter += 1
                    if processed_counter % self.log_interval == 0:
                        if success_log:
                            success_log.flush()
                        if error_log:
                            error_log.flush()
                    pbar.update(1)

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


def get_solids(shape):
    # add solids
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    solids = []
    while explorer.More():
        solids.append(explorer.Current())
        explorer.Next()
    # add shells
    explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    while explorer.More():
        solids.append(explorer.Current())
        explorer.Next()
    return solids

def sample_points_on_shape(shape, deflection=1e-1, samples_per_triangle=5):
    BRepMesh_IncrementalMesh(shape, deflection)
    points = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = exp.Current()
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation:
            # Get triangle nodes
            for t in range(1, triangulation.NbTriangles() + 1):
                tri = triangulation.Triangle(t)
                n1, n2, n3 = tri.Get()
                p1 = triangulation.Node(n1)
                p2 = triangulation.Node(n2)
                p3 = triangulation.Node(n3)
                # Sample points inside the triangle using barycentric coordinates
                for _ in range(samples_per_triangle):
                    r1 = random.random()
                    r2 = random.random()
                    # Ensure the point is inside the triangle
                    if r1 + r2 > 1:
                        r1 = 1 - r1
                        r2 = 1 - r2
                    x = p1.X() + r1 * (p2.X() - p1.X()) + r2 * (p3.X() - p1.X())
                    y = p1.Y() + r1 * (p2.Y() - p1.Y()) + r2 * (p3.Y() - p1.Y())
                    z = p1.Z() + r1 * (p2.Z() - p1.Z()) + r2 * (p3.Z() - p1.Z())
                    points.append((x, y, z))
        exp.Next()
    return points

def save_points_as_ply(points, filename):
    with open(filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for pt in points:
            f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")

def boxes_close(box1, box2, tol):
    box1.Enlarge(tol)
    box2.Enlarge(tol)
    return not box1.IsOut(box2)

def get_bbox(shape):
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    return box

def create_gp_trsf_from_matrix(matrix, scale_factor=10.0):
    """
    Converts a 4x4 transformation matrix (row-major) into a gp_Trsf object,
    applying a scaling factor to the translation components only.

    Parameters:
    - matrix (list of list of float): a 4x4 transformation matrix in row-major order
    - scale_factor (float): scale applied to the translation part (usually for unit conversion)

    Returns:
    - gp_Trsf object representing the transformation
    """
    if not (len(matrix) == 4 and all(len(row) == 4 for row in matrix)):
        raise ValueError("Input must be a 4x4 matrix.")
    m = np.array(matrix)
    trsf = gp_Trsf()
    trsf.SetValues(
        float(m[0][0]), float(m[0][1]), float(m[0][2]), float(m[0][3]) * scale_factor,
        float(m[1][0]), float(m[1][1]), float(m[1][2]), float(m[1][3]) * scale_factor,
        float(m[2][0]), float(m[2][1]), float(m[2][2]), float(m[2][3]) * scale_factor
    )

    return trsf

def load_geometry(assembly_file_path):
    """
            if occ_uuid is None:
            body_id = body_uuid
        else:
            body_id = f"{occ_uuid}_{body_uuid}"

        As defined above in assembly_graph.py. Here, body_id = node_id
    """
    try:
        ag = AssemblyGraph(assembly_file_path)
        graph = ag.get_graph_networkx()
    except Exception as e:
        raise RuntimeError(f"Failed to load assembly graph from {assembly_file_path}: {e}. Likely due to already having been processed.")

    shapes = []
    node_ids = []
    for index, (node_id, node_data) in enumerate(graph.nodes.data()):
        node_smt_file = assembly_file_path.parent / f"{node_data['body_file']}.step"
        shape = read_step_file(str(node_smt_file))
        trsf = create_gp_trsf_from_matrix(node_data['transform'], 10)
        shape = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
        shapes.append(shape)
        node_ids.append(node_id)

    solids = shapes
    assert len(solids) == len(shapes) and len(solids) == len(node_ids), "Mismatch in number of solids and nodes."
    return solids, node_ids

def get_faces(shape):
    return [f for f in TopologyExplorer(shape).faces()]


def compute_area(shape):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, props, True, True)
    return props.Mass()

def compute_volume(shape):
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return abs(props.Mass())

def face_to_body_contact(faces1, solid2, bbox2, proximity_tol=1e-3, area_tol=1e-6):
    points = []
    areas = 0.0
    volumes = 0.0
    for f1 in faces1:
        bbox_f1 = get_bbox(f1)
        if boxes_close(bbox_f1, bbox2, proximity_tol):
            common = BRepAlgoAPI_Common(f1, solid2)
            if common.IsDone() and not common.Shape().IsNull():
                area = compute_area(common.Shape())
                if area > area_tol:
                    volume = compute_volume(common.Shape())
                    point = sample_points_on_shape(common.Shape(), deflection=0.01, samples_per_triangle=10)
                    if len(point) > 0:
                        points.extend(point)
                        areas += area
                        volumes += volume
    return points, areas, volumes

def bnd_box_to_dict(box):
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return {
        "type": "BoundingBox3D",
        "min_point": {"type": "Point3D", "x": xmin, "y": ymin, "z": zmin},
        "max_point": {"type": "Point3D", "x": xmax, "y": ymax, "z": zmax},
    }

def compute_contacts(solids, node_keys, assembly_file_path, proximity_tol=1e-3, area_tol=1e-6, surface_type_source_dir: Path | None = None):
    contacts = []
    bboxes = [get_bbox(solid) for solid in solids]

    # get source for contact metadata (surface types)
    source_json_path = assembly_file_path
    if surface_type_source_dir:
        assembly_folder_name = assembly_file_path.parent.name
        candidate_path = surface_type_source_dir / assembly_folder_name / "assembly.json"
        if candidate_path.exists():
            source_json_path = candidate_path
        else:
            print(f"[Warning] Surface type source not found for {assembly_folder_name}, using target file as fallback.")
    try:
        with open(str(source_json_path), 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        original_contacts = original_data.get('contacts', [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[Warning] Could not load or parse source JSON {source_json_path}: {e}. Proceeding without surface types.")
        original_contacts = []

    # mapping from each specific face (body_id, face_index) to its surface type
    face_to_surface_type = {}
    for c in original_contacts:
        for ent_key in ('entity_one', 'entity_two'):
            ent = c.get(ent_key, {})
            body = ent.get('body')
            occ = ent.get('occurrence')
            face_idx = ent.get('index')
            surface_type = ent.get('surface_type')
            if body and face_idx is not None and surface_type:
                body_id = f"{occ}_{body}" if occ else body
                face_to_surface_type[(body_id, face_idx)] = surface_type

    for i, solid1 in enumerate(solids):
        faces1 = get_faces(solid1)
        faces_cache = {}
        for j in range(i + 1, len(solids)):
            if not boxes_close(bboxes[i], bboxes[j], proximity_tol):
                continue
            # build faces2 list lazily
            if j not in faces_cache:
                faces_cache[j] = get_faces(solids[j])

            points, area, volume = face_to_body_contact(faces1, solids[j], bboxes[j], proximity_tol, area_tol)
            if len(points) > 0:
                try:
                    body1 = node_keys[i].split("_")[1]
                    occ1 = node_keys[i].split("_")[0]
                except IndexError:
                    body1 = node_keys[i]
                    occ1 = None
                try:
                    body2 = node_keys[j].split("_")[1]
                    occ2 = node_keys[j].split("_")[0]
                except IndexError:
                    body2 = node_keys[j]
                    occ2 = None
                # get first face indices that contributed (coarse heuristic)
                face1_index = None
                face2_index = None
                for idx_f1, f1 in enumerate(faces1):
                    bbox_f1 = get_bbox(f1)
                    if not boxes_close(bbox_f1, bboxes[j], proximity_tol):
                        continue
                    # faces in solid2
                    for idx_f2, f2 in enumerate(faces_cache[j]):
                        bbox_f2 = get_bbox(f2)
                        if not boxes_close(bbox_f1, bbox_f2, proximity_tol):
                            continue
                        # quick test common
                        common_test = BRepAlgoAPI_Common(f1, f2)
                        if common_test.IsDone() and not common_test.Shape().IsNull():
                            face1_index = idx_f1
                            face2_index = idx_f2
                            break
                    if face1_index is not None:
                        break
                node_key1 = node_keys[i]
                node_key2 = node_keys[j]
                surface1 = face_to_surface_type.get((node_key1, face1_index))
                surface2 = face_to_surface_type.get((node_key2, face2_index))

                contacts.append({
                    "entity_one": {
                        "body": body1,
                        "occurrence": occ1,
                        "index": face1_index,
                        "bounding_box": bnd_box_to_dict(bboxes[i]),
                        "surface_type": surface1,
                        "point_on_entity": {"type": "Point3D", "x": None, "y": None, "z": None}
                    },
                    "entity_two": {
                        "body": body2,
                        "occurrence": occ2,
                        "index": face2_index,
                        "bounding_box": bnd_box_to_dict(bboxes[j]),
                        "surface_type": surface2,
                        "point_on_entity": {"type": "Point3D", "x": None, "y": None, "z": None}
                    },
                    "contact_area": area,
                    "contact_volume": volume,
                    "points": points,
                    "id": f"{i}_{j}",
                })

    return contacts

def save_contacts(assembly_file_path, contacts):
    with open(assembly_file_path, "r") as f:
        assembly_json = json.load(f)

    for contact in contacts:
        points = contact.get("points", None)
        if points is None or len(points) == 0:
            continue
        ply_filename = assembly_file_path.parent / f"contact/contact_{contact['id']}.ply"
        ply_filename.parent.mkdir(parents=True, exist_ok=True)
        save_points_as_ply(points, ply_filename)
        if "points" in contact:
            del contact["points"]
    assembly_json["contacts"] = contacts
    with open(assembly_file_path, "w") as f:
        json.dump(assembly_json, f, indent=4)

def display_geometry(solids):
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(solids, update=True)
    start_display()

def generate_assembly_contacts(assembly_file_path, surface_type_source_dir: Path | None = None):
    try:
        solids, node_keys = load_geometry(assembly_file_path)
        contacts = compute_contacts(solids, node_keys, assembly_file_path, proximity_tol=0.1, area_tol=0.01, surface_type_source_dir=surface_type_source_dir)
        save_contacts(assembly_file_path, contacts)
        return f"Number of generated contacts: {len(contacts)}"
    except Exception as e:
        raise e

def generate_assembly_contacts_worker(assembly_file_path):
    """Worker function for multiprocessing that calls generate_assembly_contacts with no surface_type_source_dir."""
    return generate_assembly_contacts(assembly_file_path, surface_type_source_dir=None)

def generate_assembly_contacts_worker_with_source(args):
    """Worker function for multiprocessing that accepts (assembly_file_path, surface_type_source_dir) tuple."""
    assembly_file_path, surface_type_source_dir = args
    return generate_assembly_contacts(assembly_file_path, surface_type_source_dir=surface_type_source_dir)

def augment_existing_contacts_worker(args):
    """Worker function for multiprocessing that accepts (assembly_file_path, surface_type_source_dir) tuple."""
    assembly_file_path, surface_type_source_dir = args
    return augment_existing_contacts(assembly_file_path, surface_type_source_dir=surface_type_source_dir)

def augment_existing_contacts_worker_simple(assembly_file_path):
    """Worker function for multiprocessing that calls augment_existing_contacts with no surface_type_source_dir."""
    return augment_existing_contacts(assembly_file_path, surface_type_source_dir=None)

def parallel_test(assembly_dir, num_assemblies=100, logs_dir: Path | None = None, max_workers: int | None = None, log_interval: int = 300, augment_only: bool = False, surface_type_source_dir: Path | None = None, force: bool = False):
    assembly_files = list(assembly_dir.glob("*/assembly.json"))
    print(f"Found {len(assembly_files)} assembly files.")
    skipped_files = []
    assembly_files_to_process = []
    for af in assembly_files:
        contact_dir = af.parent / "contact"
        has_contacts = contact_dir.exists() and any(contact_dir.iterdir())
        if augment_only or force:
            assembly_files_to_process.append(af)
        else:
            if has_contacts:
                skipped_files.append(af)
            else:
                assembly_files_to_process.append(af)
    if skipped_files:
        print(f"Skipping {len(skipped_files)} assemblies that already contain contacts (see skipped log). Use --force to recalculate.")

    if logs_dir is None:
        logs_dir = assembly_dir
    else:
        logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    success_log = logs_dir / f"contacts_success_{timestamp}.log"
    error_log = logs_dir / f"contacts_errors_{timestamp}.log"
    skipped_log = logs_dir / f"contacts_skipped_{timestamp}.log"

    if skipped_files:
        with open(skipped_log, "w", encoding="utf-8") as sk_log:
            for item in skipped_files:
                sk_log.write(f"{item} | already contains contacts\n")
        print(f"\n⏭️ Skipped Assemblies ({len(skipped_files)}):")
        for item in skipped_files:
            print(f"  Skipped: {item}")

    executor_timeout = 360  # 6 mins per assembly
    if surface_type_source_dir is None:
        if augment_only:
            worker_func = augment_existing_contacts_worker_simple
        else:
            worker_func = generate_assembly_contacts_worker
        items_to_process = assembly_files_to_process[:num_assemblies]
    else:
        if augment_only:
            worker_func = augment_existing_contacts_worker
        else:
            worker_func = generate_assembly_contacts_worker_with_source
        items_to_process = [(p, surface_type_source_dir) for p in assembly_files_to_process[:num_assemblies]]

    executor = ParallelExecutor(worker_func, max_workers=max_workers, log_interval=log_interval, timeout=executor_timeout, desc="Augmenting" if augment_only else "Processing assemblies")
    results, errors = executor.run(items_to_process, success_log, error_log)

    print("\n✅ Successful assemblies:", len(results))
    print("⚠️ Assemblies with errors:", len(errors))

def single_assembly_test(assembly_dir):
    assembly_file_path = assembly_dir / "NAME_OF_ASSEMBLT/assembly.json"    # cylinder?
    solids, node_keys = load_geometry(assembly_file_path)
    display_geometry(solids)
    contacts = compute_contacts(solids, node_keys, assembly_file_path, proximity_tol=0.1, area_tol=0.01)
    print(f"len contacts:\n{len(contacts)}")
    for idx, contact in enumerate(contacts):
            print(f"Contact {idx}: area = {contact['contact_area']}")
            print(f"Contact {idx}: volume = {contact['contact_volume']}")
    save_contacts(assembly_file_path, contacts)

def debug_augment_one(assembly_folder: Path):
    """Load one assembly, regenerate contacts with enrichment, and print a preview."""
    assembly_json = assembly_folder / "assembly.json"
    if not assembly_json.exists():
        print(f"[DEBUG] assembly.json not found in {assembly_folder}")
        return

    print(f"[DEBUG] Regenerating contacts for {assembly_folder.name} ...")
    try:
        solids, node_keys = load_geometry(assembly_json)
        contacts = compute_contacts(solids, node_keys, assembly_json, proximity_tol=0.1, area_tol=0.01)
        print(f"[DEBUG] computed {len(contacts)} contacts. Preview (first item):\n")
        ans = input("Write enriched contacts back to assembly.json? [y/N]: ").lower()
        if ans == "y":
            save_contacts(assembly_json, contacts)
            print("[DEBUG] assembly.json updated.")
    except Exception as exc:
        print(f"[DEBUG] Error: {exc}")
        raise

def augment_existing_contacts(assembly_json_path: Path, surface_type_source_dir: Path | None = None):
    try:
        with assembly_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        contacts = data.get("contacts") or []
        if not contacts:
            return f"{assembly_json_path} | no contacts"

        source_json_path = assembly_json_path
        if surface_type_source_dir:
            assembly_folder_name = assembly_json_path.parent.name
            candidate_path = surface_type_source_dir / assembly_folder_name / "assembly.json"
            if candidate_path.exists():
                source_json_path = candidate_path
            else:
                print(f"[Warning] Augment: Surface type source not found for {assembly_folder_name}, using target file as fallback.")

        face_to_surface_type = {}
        try:
            with open(str(source_json_path), 'r', encoding='utf-8') as f:
                source_data = json.load(f)
            source_contacts = source_data.get('contacts') or []
            for c in source_contacts:
                for ent_key in ('entity_one', 'entity_two'):
                    ent = c.get(ent_key, {})
                    body = ent.get('body')
                    occ = ent.get('occurrence')
                    face_idx = ent.get('index')
                    surface_type = ent.get('surface_type')
                    if body and face_idx is not None and surface_type:
                        body_id = f"{occ}_{body}" if occ else body
                        face_to_surface_type[(body_id, face_idx)] = surface_type
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[Warning] Could not load or parse source JSON {source_json_path}: {e}. Proceeding without updating surface types.")

        ag = AssemblyGraph(assembly_json_path)
        graph = ag.get_graph_networkx()
        solids = {}
        bboxes = {}
        faces_cache = {}
        for node_id, node_data in graph.nodes.data():
            step_file = assembly_json_path.parent / f"{node_data['body_file']}.step"
            shape = read_step_file(str(step_file))
            trsf = create_gp_trsf_from_matrix(node_data["transform"], 10)
            shape = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
            solids[node_id] = shape
            box = Bnd_Box(); brepbndlib.Add(shape, box)
            bboxes[node_id] = bnd_box_to_dict(box)
            faces_cache[node_id] = get_faces(shape)

        placeholder_pt = {"type": "Point3D", "x": None, "y": None, "z": None}
        for c in contacts:
            if 'surface_type' in c:
                del c['surface_type']
            # check face indices exist, compute them if necessary
            id1 = c["entity_one"]["body"] if c["entity_one"].get("occurrence") is None else f"{c['entity_one']['occurrence']}_{c['entity_one']['body']}"
            id2 = c["entity_two"]["body"] if c["entity_two"].get("occurrence") is None else f"{c['entity_two']['occurrence']}_{c['entity_two']['body']}"
            
            if c["entity_one"].get("index") is None:
                idx1 = idx2 = None
                for i, f1 in enumerate(faces_cache[id1]):
                    for j, f2 in enumerate(faces_cache[id2]):
                        common = BRepAlgoAPI_Common(f1, f2)
                        if common.IsDone() and not common.Shape().IsNull():
                            idx1, idx2 = i, j; break
                    if idx1 is not None:
                        break
                c["entity_one"]["index"] = idx1
                c["entity_two"]["index"] = idx2

            for ent_key in ("entity_one", "entity_two"):
                ent = c[ent_key]
                body_id = ent["body"] if ent.get("occurrence") is None else f"{ent['occurrence']}_{ent['body']}"
                face_idx = ent.get("index")
                if body_id in bboxes:
                    ent["bounding_box"] = bboxes[body_id]
                ent.setdefault("point_on_entity", placeholder_pt)
                
                if face_idx is not None:
                    lookup_key = (body_id, face_idx)
                    if lookup_key in face_to_surface_type:
                        ent["surface_type"] = face_to_surface_type[lookup_key]
        with assembly_json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return f"{assembly_json_path} | augmented"
    except Exception as e:
        return f"{assembly_json_path} | ERROR: {e}"

def process_rerun_list(rerun_file_path: Path, max_workers: int | None = None, log_interval: int = 100, surface_type_source_dir: Path | None = None, logs_dir: Path | None = None):
    """
    Process assemblies from a rerun list file (like comprehensive_rerun_list.txt).
    """
    try:
        with rerun_file_path.open("r", encoding="utf-8") as f:
            assembly_paths = [Path(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"[ERROR] Rerun list file not found at: {rerun_file_path}")
        return

    print(f"Found {len(assembly_paths)} assemblies to reprocess from {rerun_file_path}")
    
    if not assembly_paths:
        print("No assemblies to process!")
        return

    if logs_dir is None:
        logs_dir = rerun_file_path.parent
    else:
        logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    success_log = logs_dir / f"rerun_contacts_success_{timestamp}.log"
    error_log = logs_dir / f"rerun_contacts_errors_{timestamp}.log"
    
    print(f"Logs will be saved to:")
    print(f"  Success: {success_log}")
    print(f"  Errors: {error_log}")

    print(f"\nStarting reprocessing of {len(assembly_paths)} assemblies...")
    executor_timeout = 600  # 10 minutes per assembly

    if surface_type_source_dir:
        print(f"[WARNING] surface_type_source_dir is not supported in rerun mode yet. Processing without it.")
    
    worker_func = generate_assembly_contacts_worker

    executor = ParallelExecutor(
        worker_func, 
        max_workers=max_workers, 
        log_interval=log_interval, 
        timeout=executor_timeout, 
        desc="Reprocessing assemblies"
    )
    
    results, errors = executor.run(assembly_paths, success_log, error_log)

    print(f"\n=== REPROCESSING COMPLETE ===")
    print(f"✅ Successfully reprocessed: {len(results)}")
    print(f"❌ Failed assemblies: {len(errors)}")
    
    if errors:
        print(f"\nFailed assemblies:")
        for assembly_path, error in list(errors.items())[:10]:
            print(f"  {assembly_path.parent.name}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more (see error log)")

def comprehensive_rerun_check(success_log_path=None, error_log_path=None, assembly_dir=None):
    """
    Comprehensive check to determine which assemblies need rerunning by examining:
    1. Success log entries vs actual PLY files
    2. Error log entries (timeouts, failures)
    3. Assemblies with no processing logs at all
    4. Assemblies with incomplete contact generation
    """
    
    if not any([success_log_path, error_log_path, assembly_dir]):
        print("[ERROR] Must provide at least one of: success log, error log, or assembly directory")
        return
    
    assemblies_needing_rerun = set()
    successfully_processed = set()
    error_reasons = {}
    
    if success_log_path and success_log_path.exists():
        print(f"Processing success log: {success_log_path}")
        with success_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                if " | Number of generated contacts: " in line:
                    parts = line.strip().split(" | Number of generated contacts: ")
                    if len(parts) == 2:
                        assembly_path = Path(parts[0])
                        try:
                            reported_contacts = int(parts[1])
                        except ValueError:
                            continue
                        
                        contact_folder = assembly_path.parent / "contact"
                        if contact_folder.exists():
                            ply_files = list(contact_folder.glob("*.ply"))
                            actual_contacts = len(ply_files)
                            
                            if actual_contacts == reported_contacts and reported_contacts > 0:
                                successfully_processed.add(assembly_path)
                                print(f"✅ {assembly_path.parent.name}: {actual_contacts} contacts complete")
                            else:
                                assemblies_needing_rerun.add(assembly_path)
                                reason = f"PLY mismatch: reported {reported_contacts}, found {actual_contacts}"
                                error_reasons[assembly_path] = reason
                                print(f"⚠️ {assembly_path.parent.name}: {reason}")
                        else:
                            assemblies_needing_rerun.add(assembly_path)
                            error_reasons[assembly_path] = "No contact folder found"
                            print(f"⚠️ {assembly_path.parent.name}: No contact folder")
    
    if error_log_path and error_log_path.exists():
        print(f"\nProcessing error log: {error_log_path}")
        with error_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                if " | " in line:
                    parts = line.strip().split(" | ", 1)
                    if len(parts) == 2:
                        assembly_path = Path(parts[0])
                        error_reason = parts[1]
                        if assembly_path not in successfully_processed:
                            assemblies_needing_rerun.add(assembly_path)
                            error_reasons[assembly_path] = error_reason
                            print(f"❌ {assembly_path.parent.name}: {error_reason}")
    
    if assembly_dir and assembly_dir.exists():
        print(f"\nScanning assembly directory: {assembly_dir}")
        all_assemblies = set(assembly_dir.glob("*/assembly.json"))
        processed_assemblies = successfully_processed | assemblies_needing_rerun
        unprocessed = all_assemblies - processed_assemblies
        for assembly_path in unprocessed:
            contact_folder = assembly_path.parent / "contact"
            if not contact_folder.exists() or not any(contact_folder.glob("*.ply")):
                assemblies_needing_rerun.add(assembly_path)
                error_reasons[assembly_path] = "Never processed"
                print(f"🔄 {assembly_path.parent.name}: Never processed")
    
    print(f"\n=== COMPREHENSIVE RERUN ANALYSIS ===")
    print(f"✅ Successfully processed assemblies: {len(successfully_processed)}")
    print(f"⚠️ Assemblies needing rerun: {len(assemblies_needing_rerun)}")
    
    reason_counts = {}
    for reason in error_reasons.values():
        # Simplify reason for categorization
        if "Timeout" in reason:
            category = "Timeout"
        elif "PLY mismatch" in reason:
            category = "Incomplete contacts"
        elif "Failed to load" in reason:
            category = "Loading errors"
        elif "Never processed" in reason:
            category = "Unprocessed"
        elif "No contact" in reason:
            category = "Missing output"
        else:
            category = "Other errors"
        reason_counts[category] = reason_counts.get(category, 0) + 1
    
    print("\n📊 Breakdown of issues:")
    for category, count in sorted(reason_counts.items()):
        print(f"   {category}: {count}")
    
    if assemblies_needing_rerun:
        rerun_file = Path("comprehensive_rerun_list.txt")
        detailed_file = Path("rerun_reasons.txt")
        print(f"\n📝 Writing {len(assemblies_needing_rerun)} assemblies to {rerun_file}")
        with rerun_file.open("w", encoding="utf-8") as f:
            for assembly_path in sorted(assemblies_needing_rerun):
                f.write(f"{assembly_path}\n")
        print(f"📝 Writing detailed reasons to {detailed_file}")
        with detailed_file.open("w", encoding="utf-8") as f:
            for assembly_path in sorted(assemblies_needing_rerun):
                reason = error_reasons.get(assembly_path, "Unknown")
                f.write(f"{assembly_path} | {reason}\n")

def check_contact_completeness(log_file_path: Path):
    """
    Simple check: for each assembly in the log, verify that the number of 
    reported contacts matches the number of .ply files in the contact/ folder.
    """
    try:
        with log_file_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[ERROR] Log file not found at: {log_file_path}")
        return

    assemblies_to_rerun = []
    complete_assemblies = []
    
    for line in lines:
        if " | Number of generated contacts: " in line:
            parts = line.strip().split(" | Number of generated contacts: ")
            if len(parts) == 2:
                assembly_path = Path(parts[0])
                try:
                    reported_contacts = int(parts[1])
                except ValueError:
                    print(f"[Warning] Could not parse contact count from: {line.strip()}")
                    continue                
                contact_folder = assembly_path.parent / "contact"
                if contact_folder.exists():
                    ply_files = list(contact_folder.glob("*.ply"))
                    actual_contacts = len(ply_files)
                    
                    if actual_contacts != reported_contacts:
                        print(f"[MISMATCH] {assembly_path.parent.name}: reported {reported_contacts}, found {actual_contacts} ply files")
                        assemblies_to_rerun.append(assembly_path)
                    else:
                        complete_assemblies.append(assembly_path)
                else:
                    print(f"[MISSING FOLDER] {assembly_path.parent.name}: no contact folder found")
                    assemblies_to_rerun.append(assembly_path)
    
    print(f"\n--- Contact Completeness Check ---")
    print(f"Complete assemblies (contacts match ply files): {len(complete_assemblies)}")
    print(f"Assemblies needing rerun (mismatch or missing): {len(assemblies_to_rerun)}")
    
    if assemblies_to_rerun:
        rerun_file = Path("assemblies_to_rerun.txt")
        print(f"\nWriting {len(assemblies_to_rerun)} assemblies to {rerun_file}")
        with rerun_file.open("w", encoding="utf-8") as f:
            for p in assemblies_to_rerun:
                f.write(f"{p}\n")

def check_timed_out_assemblies(log_file_path: Path):
    """
    Parses a log file of timed-out assemblies, checks for missing data,
    and prepares a list of assemblies to rerun.
    """
    try:
        with log_file_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[ERROR] Log file not found at: {log_file_path}")
        return

    assembly_paths = []
    for line in lines:
        if " | " in line:
            path_str = line.split(" | ")[0].strip()
            assembly_paths.append(Path(path_str))

    print(f"Found {len(assembly_paths)} timed-out assemblies in {log_file_path}.")
    print("Checking for missing component files...")
    assemblies_with_missing_files = []
    assemblies_to_rerun = []
    for assembly_json in tqdm(assembly_paths, desc="Checking Assemblies"):
        if not assembly_json.exists():
            print(f"\n[Warning] assembly.json not found, skipping: {assembly_json}")
            continue
        if assembly_json.stat().st_size == 0:
            print(f"\n[Error] Skipping empty assembly file: {assembly_json}")
            continue
        try:
            ag = AssemblyGraph(assembly_json)
            graph = ag.get_graph_networkx()
            is_missing_files = False
            for _, node_data in graph.nodes.data():
                step_file = assembly_json.parent / f"{node_data['body_file']}.step"
                if not step_file.exists():
                    print(f"\n[Missing File] for {assembly_json.parent.name}: {step_file.name}")
                    is_missing_files = True
            if is_missing_files:
                assemblies_with_missing_files.append(assembly_json)
            else:
                assemblies_to_rerun.append(assembly_json)

        except Exception as e:
            print(f"\n[Error] Could not process {assembly_json}: {e!r}") # Using !r to get a more detailed representation
            continue
    print("\n--- Check Complete ---")
    print(f"Total timed-out assemblies checked: {len(assembly_paths)}")
    print(f"Assemblies with missing component files: {len(assemblies_with_missing_files)}")
    print(f"Assemblies with all files present (likely timeout due to complexity): {len(assemblies_to_rerun)}")
    
    rerun_file = Path("rerun_these.txt")
    print(f"\nWriting {len(assemblies_to_rerun)} assemblies to {rerun_file} for reprocessing.")
    with rerun_file.open("w", encoding="utf-8") as f:
        for p in assemblies_to_rerun:
            f.write(f"{p}\n")

def create_smart_rerun_strategy(error_log_path: Path, max_attempts: int = 3):
    """
    Smarter rerun strategy that tracks failure history and stops trying assemblies
    that consistently fail with the same errors.
    """
    if not error_log_path.exists():
        print(f"[ERROR] Log file not found at: {error_log_path}")
        return

    print(f"Analyzing error log with smart retry strategy: {error_log_path}")    
    failure_history_file = Path("failure_history.json")
    failure_history = {}
    if failure_history_file.exists():
        with failure_history_file.open("r") as f:
            failure_history = json.load(f)
    
    timeout_errors = set()
    transient_errors = set()
    permanent_errors = set()
    
    with error_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if " | " in line:
                path_str, reason = line.strip().split(" | ", 1)
                assembly_path = str(Path(path_str))                
                if assembly_path not in failure_history:
                    failure_history[assembly_path] = {"attempts": 0, "errors": []}
                
                failure_history[assembly_path]["attempts"] += 1
                failure_history[assembly_path]["errors"].append(reason)
                
                if "Timeout after" in reason:
                    if failure_history[assembly_path]["attempts"] < max_attempts:
                        timeout_errors.add(Path(path_str))
                elif "'NoneType' object is not iterable" in reason or "terminated abruptly" in reason:
                    if failure_history[assembly_path]["attempts"] < max_attempts:
                        transient_errors.add(Path(path_str))
                else:
                    permanent_errors.add(Path(path_str))
    with failure_history_file.open("w") as f:
        json.dump(failure_history, f, indent=2)
    assemblies_to_retry = timeout_errors | transient_errors
    
    print(f"\n=== SMART RETRY ANALYSIS ===")
    print(f"Timeout errors (will retry with longer timeout): {len(timeout_errors)}")
    print(f"Transient errors (will retry): {len(transient_errors)}")
    print(f"Permanent errors (will NOT retry): {len(permanent_errors)}")
    print(f"Total assemblies to retry: {len(assemblies_to_retry)}")
    
    if permanent_errors:
        print(f"\nExamples of assemblies we're giving up on:")
        for i, assembly in enumerate(list(permanent_errors)[:5]):
            history = failure_history.get(str(assembly), {})
            attempts = history.get("attempts", 0)
            last_error = history.get("errors", ["Unknown"])[-1]
            print(f"  {assembly.parent.name}: {attempts} attempts, last error: {last_error}")
    
    if timeout_errors:
        timeout_file = Path("timeout_retry_list.txt")
        print(f"\nWriting {len(timeout_errors)} timeout assemblies to {timeout_file}")
        with timeout_file.open("w") as f:
            for path in sorted(timeout_errors):
                f.write(f"{path}\n")
    
    if transient_errors:
        transient_file = Path("transient_retry_list.txt")
        print(f"Writing {len(transient_errors)} transient error assemblies to {transient_file}")
        with transient_file.open("w") as f:
            for path in sorted(transient_errors):
                f.write(f"{path}\n")
    
    if assemblies_to_retry:
        combined_file = Path("smart_retry_list.txt")
        print(f"Writing {len(assemblies_to_retry)} total assemblies to {combined_file}")
        with combined_file.open("w") as f:
            for path in sorted(assemblies_to_retry):
                f.write(f"{path}\n")
    
    return len(assemblies_to_retry)

def create_rerun_list_from_errors(error_log_path: Path):
    """
    Analyze error log and create a rerun list for recoverable errors only.
    Focuses on timeouts, NoneType errors, and process terminations.
    """
    if not error_log_path.exists():
        print(f"[ERROR] Log file not found at: {error_log_path}")
        return
    
    print(f"Analyzing error log for recoverable errors: {error_log_path}")
    
    recoverable_assemblies = set()
    error_counts = {}
    
    with error_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if " | " in line:
                path_str, reason = line.strip().split(" | ", 1)
                assembly_path = Path(path_str)                
                if "Timeout after" in reason:
                    category = "Timeout"
                elif "'NoneType' object is not iterable" in reason:
                    category = "NoneType error"
                elif "terminated abruptly" in reason:
                    category = "Process terminated"
                elif "No contacts found" in reason:
                    category = "No contacts found"
                elif "Failed to load" in reason:
                    category = "Loading error"
                else:
                    category = "Other error"
                
                error_counts[category] = error_counts.get(category, 0) + 1
                
                if category in ["Timeout", "NoneType error", "Process terminated"]:
                    recoverable_assemblies.add(assembly_path)
    
    print(f"\n=== ERROR ANALYSIS ===")
    for category, count in sorted(error_counts.items()):
        recoverable = "✓" if category in ["Timeout", "NoneType error", "Process terminated"] else "✗"
        print(f"  {category}: {count} {recoverable}")
    
    print(f"\nRecoverable assemblies to retry: {len(recoverable_assemblies)}")
    
    if recoverable_assemblies:
        rerun_file = Path("rerun_round2_list.txt")
        print(f"Writing {len(recoverable_assemblies)} assemblies to {rerun_file}")
        with rerun_file.open("w", encoding="utf-8") as f:
            for assembly_path in sorted(recoverable_assemblies):
                f.write(f"{assembly_path}\n")
    else:
        print("No recoverable errors found - no rerun list created.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate or debug contacts")
    parser.add_argument("--source", type=str, default=None, help="Root assembly directory for batch mode")
    parser.add_argument("--check-timeouts", type=str, default=None, help="Path to a log file with timed out assemblies to check.")
    parser.add_argument("--check-completeness", type=str, default=None, help="Path to a success log file to check if contacts match ply files.")
    parser.add_argument("--comprehensive-check", action="store_true", help="Comprehensive check using success log, error log, and assembly directory.")
    parser.add_argument("--success-log", type=str, help="Path to success log file for comprehensive check.")
    parser.add_argument("--error-log", type=str, help="Path to error log file for comprehensive check.")
    parser.add_argument("--rerun-from-file", type=str, help="Path to a file containing list of assemblies to reprocess (e.g., comprehensive_rerun_list.txt).")
    parser.add_argument("--analyze-rerun-log", type=str, help="Path to a rerun error log to analyze for creating a new rerun list for timeouts and crashes.")
    parser.add_argument("--debug", type=str, help="Path to single assembly folder for debug augmentation")
    parser.add_argument("--num", type=int, default=100, help="Number of assemblies to process in batch mode")
    parser.add_argument("--workers", type=int, default=4, help="Worker processes in batch mode")
    parser.add_argument("--logs", type=str, default=None, help="Directory to store logs")
    parser.add_argument("--st-source", type=str, default=None, help="Directory with original assemblies to source surface_type from.")
    parser.add_argument("--augment", action="store_true", help="Only augment existing contacts (bounding boxes etc.)")
    parser.add_argument("--force", action="store_true", help="Force recalculate contacts even if they already exist")
    parser.add_argument("--smart-retry", type=str, help="Path to error log for smart retry analysis (tracks failure history)")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum retry attempts per assembly")
    args = parser.parse_args()

    if args.analyze_rerun_log:
        create_rerun_list_from_errors(Path(args.analyze_rerun_log))
        exit()

    if args.rerun_from_file:
        st_source_path = Path(args.st_source) if args.st_source else None
        logs_path = Path(args.logs) if args.logs else None
        process_rerun_list(
            rerun_file_path=Path(args.rerun_from_file),
            max_workers=args.workers,
            log_interval=100,
            surface_type_source_dir=st_source_path,
            logs_dir=logs_path
        )
        exit()

    if args.comprehensive_check:
        success_log = Path(args.success_log) if args.success_log else None
        error_log = Path(args.error_log) if args.error_log else None
        assembly_dir = Path(args.source) if args.source else None
        comprehensive_rerun_check(success_log, error_log, assembly_dir)
        exit()

    if args.check_completeness:
        check_contact_completeness(Path(args.check_completeness))
        exit()

    if args.check_timeouts:
        check_timed_out_assemblies(Path(args.check_timeouts))
        exit()

    if args.debug:
        debug_augment_one(Path(args.debug))
        exit()

    if args.source is None:
        print("[ERROR] Please provide --source for batch mode or --debug for single assembly test")
        exit(1)

    logs_path = Path(args.logs) if args.logs else None
    st_source_path = Path(args.st_source) if args.st_source else None
    parallel_test(
        assembly_dir=Path(args.source),
        num_assemblies=args.num,
        logs_dir=logs_path,
        max_workers=args.workers,
        log_interval=100,
        augment_only=args.augment,
        surface_type_source_dir=st_source_path,
        force=args.force,
    )

    if args.smart_retry:
        remaining = create_smart_rerun_strategy(Path(args.smart_retry), args.max_attempts)
        if remaining == 0:
            print("\n🎉 NO MORE ASSEMBLIES TO RETRY! You're done!")
        exit()
