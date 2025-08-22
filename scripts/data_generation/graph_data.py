import os
import ast
import json
import torch
import random
import joblib
import copy
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from torch_geometric.data import Data, InMemoryDataset
from .assemblyGraphGeneration.assembly_graph import AssemblyGraph

def load_ply_as_points(ply_path, max_points=1024):
    """
    Load a .ply file and return points as numpy array.
    Returns (N, 3) array where N <= max_points.
    """
    try:
        import trimesh
        mesh = trimesh.load(ply_path)
        
        if hasattr(mesh, 'vertices'):
            points = mesh.vertices
        elif hasattr(mesh, 'points'):
            points = mesh.points
        else:
            # Try to extract points from faces if it's a mesh
            if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                points = mesh.vertices
            else:
                logging.warning("Could not extract points from %s", ply_path)
                return np.zeros((max_points, 3), dtype=np.float32)
        
        # Subsample if too many points
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
        elif len(points) < max_points:
            # Pad with zeros or repeat points
            padding_needed = max_points - len(points)
            if len(points) > 0:
                # Repeat existing points to fill
                repeat_indices = np.random.choice(len(points), padding_needed, replace=True)
                padding_points = points[repeat_indices]
                points = np.vstack([points, padding_points])
            else:
                points = np.zeros((max_points, 3), dtype=np.float32)
        
        return points.astype(np.float32)
    
    except Exception as e:
        logging.error("Error loading %s: %s", ply_path, e)
        return np.zeros((max_points, 3), dtype=np.float32)


class AssemblyGraphDataset(InMemoryDataset):
    def __init__(self, root, model_type, valid_assemblies_path, embeddings_path,
                 contacts_path=None, contact_embeddings_path=None, transform=None, aug_type: str = "base",
                 random_edges: bool = False, edge_feature_mode: str = 'embedding', num_clusters: int = 1000, data_id: str = None, aug_fraction: float = 0.1):
        self.valid_assemblies_path = valid_assemblies_path
        self.embeddings_path = embeddings_path
        self.contacts_path = contacts_path
        self.contact_embeddings_path = contact_embeddings_path
        self.model_type = model_type
        self.edge_feature_mode = edge_feature_mode
        self.num_clusters = num_clusters
        self.data_id = data_id
        self.embedding_suffix = os.path.basename(self.embeddings_path).replace('.parquet', '')
        
        # load contact embeddings once at initialization for efficiency
        if contact_embeddings_path:
            logging.info("Loading contact embeddings from: %s", contact_embeddings_path)
            self.contact_embeddings_map = self._load_all_contact_embeddings()
            logging.info("finished loading contact embeddings.")

            # detect expected dimension
            expected_dim = None
            for asm_dict in self.contact_embeddings_map.values():
                for emb in asm_dict.values():
                    if emb is not None:
                        expected_dim = len(emb)
                        break
                if expected_dim is not None:
                    break
            if expected_dim is None:
                raise ValueError("All contact embeddings are empty!")

            self.contact_embedding_dim = expected_dim
            logging.info("Detected contact-embedding dim: %d", expected_dim)

        else:
            self.contact_embeddings_map = {}
            self.contact_embedding_dim = 256

        logging.info("Loaded contact embeddings for %d assemblies.", len(self.contact_embeddings_map))
        
        # determine edge attribute configuration
        self.use_scalar_contacts = contacts_path is not None
        self.use_contact_embeddings = contact_embeddings_path is not None and self.edge_feature_mode == 'embedding'

        if self.edge_feature_mode == 'one_hot' and self.contacts_path:
            self._prepare_one_hot_encoding_data()
        
        logging.info("Edge attribute configuration:")
        logging.info(f"  - Mode: {self.edge_feature_mode}")
        logging.info(f"  - Scalar contacts: {self.use_scalar_contacts}")
        logging.info(f"  - Contact embeddings: {self.use_contact_embeddings}")
        self.aug_type = aug_type
        self.random_edges = (aug_type == "RE") or random_edges
        self.aug_fraction = aug_fraction

        logging.info("valid_assemblies_path: %s", self.valid_assemblies_path)
        logging.info("embeddings_path: %s", self.embeddings_path)
        if contacts_path:
            logging.info("contacts_path: %s", self.contacts_path)
        if contact_embeddings_path:
            logging.info("contact_embeddings_path: %s", self.contact_embeddings_path)
        
        super(AssemblyGraphDataset, self).__init__(root, transform=transform)
        base_pt = self.processed_paths[0]

        # build BASE dataset
        if not os.path.isfile(base_pt):
            logging.info("Base processed data not found – generating it now …")
            if self.model_type == 'GATv2':
                self.process_data()
            elif self.model_type == 'GATv2NoEdgeAttr':
                self.process_data_no_edge_attr()
            elif self.model_type == 'GATv2Classification' or self.model_type == 'GATClassification':
                self.process_data_classification()
            elif self.model_type == 'GATv2ClassificationNoEdgeAttr':
                self.process_data_classification_no_edge_attr()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

        # load requested augmented set
        if self.aug_type != "base":
            if self.aug_type == "PARCON":
                aug_tag = f"_{self.aug_type}_{int(self.aug_fraction * 100)}.pt"
                aug_pt = base_pt.replace(".pt", aug_tag)
            else:
                aug_pt = base_pt.replace(".pt", f"_{self.aug_type}.pt")
            if not os.path.isfile(aug_pt):
                logging.info("%s not found – running augmentation …", aug_pt)
                self.process_data_augmentation()
            logging.info("Loading augmented data from %s", aug_pt)
            self.data, self.slices = torch.load(aug_pt, weights_only=False)
        else:
            logging.info("Loading base data from %s", base_pt)
            self.data, self.slices = torch.load(base_pt, weights_only=False)
        logging.info("Loaded dataset has %d graphs", self.len())

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    def _prepare_one_hot_encoding_data(self):
        """Scans the contacts parquet file to prepare for one-hot encoding."""
        logging.info("Preparing for one-hot encoding by scanning contact data...")
        contacts_df = pd.read_parquet(self.contacts_path)
        self.contact_type_categories = contacts_df['contact_type'].unique().tolist()
        self.surface_1_categories = contacts_df['surface_1_type'].unique().tolist()
        self.surface_2_categories = contacts_df['surface_2_type'].unique().tolist()
        self.contact_type_map = {cat: i for i, cat in enumerate(self.contact_type_categories)}
        self.surface_1_map = {cat: i for i, cat in enumerate(self.surface_1_categories)}
        self.surface_2_map = {cat: i for i, cat in enumerate(self.surface_2_categories)}
        self.contact_area_mean = contacts_df['contact_area'].mean()
        self.contact_area_std = contacts_df['contact_area'].std()
        logging.info(f"Found {len(self.contact_type_categories)} unique contact types.")
        logging.info(f"Found {len(self.surface_1_categories)} unique surface 1 types.")
        logging.info(f"Found {len(self.surface_2_categories)} unique surface 2 types.")
        logging.info(f"Contact area stats: mean={self.contact_area_mean:.4f}, std={self.contact_area_std:.4f}")

    def _load_all_contact_embeddings(self):
        """Load and index all contact embeddings once at startup, filtered by valid assemblies."""
        try:
            valid_assemblies_df = pd.read_parquet(self.valid_assemblies_path)
            valid_assembly_ids = set(valid_assemblies_df['assembly_id'])
            logging.info("Loaded %d valid assembly IDs for filtering contact embeddings.", len(valid_assembly_ids))
            df = pd.read_parquet(self.contact_embeddings_path)
            df = self.convert_embedding(df)
            
            # create nested dict: {assembly_id: {contact_name: embedding}}
            embeddings_map = {}
            original_rows = len(df)
            
            # filter DataFrame before iterating
            df = df[df['assembly_ID'].isin(valid_assembly_ids)]
            
            for _, row in tqdm(df.iterrows(), desc="Loading contact embeddings", total=len(df)):
                assembly_id = row['assembly_ID']
                contact_name = row['contact_name']
                embedding = row['embedding']
                
                if assembly_id not in embeddings_map:
                    embeddings_map[assembly_id] = {}
                embeddings_map[assembly_id][contact_name] = embedding
            
            filtered_rows = len(df)
            logging.info("Loaded contact embeddings for %d assemblies.", len(embeddings_map))
            logging.info("Filtered contact embeddings from %d to %d based on valid assemblies.", original_rows, filtered_rows)
            return embeddings_map
        except Exception as e:
            logging.error("Error loading contact embeddings: %s", e)
            return {}

    def load_contact_embeddings(self, assembly_id):
        """Load contact embeddings for a specific assembly."""
        return self.contact_embeddings_map.get(assembly_id, {})
    
    def generate_random_edges(self, num_nodes, num_edges):
        """
        Generate a random connected graph with bidirectional edges, ensuring the
        total number of directed edges matches `num_edges`.
        """
        if num_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long)

        # aim for num_edges / 2 unique pairs to make bidirectional edges.
        target_undirected_edges = num_edges // 2

        # create a random spanning tree to ensure connectivity.
        nodes = list(range(num_nodes))
        random.shuffle(nodes)
        undirected_edges = set()
        for i in range(1, num_nodes):
            src = nodes[i]
            tgt = random.choice(nodes[:i])
            # Store edge in a canonical form (min_node, max_node) to represent an undirected edge.
            u, v = min(src, tgt), max(src, tgt)
            undirected_edges.add((u, v))

        # add extra random undirected edges until we reach the target count.
        max_possible_edges = num_nodes * (num_nodes - 1) // 2
        while len(undirected_edges) < target_undirected_edges and len(undirected_edges) < max_possible_edges:
            src = random.randint(0, num_nodes - 1)
            tgt = random.randint(0, num_nodes - 1)

            if src == tgt:
                continue

            # use canonical form to check for existence, avoiding duplicates like (1, 2) and (2, 1).
            u, v = min(src, tgt), max(src, tgt)
            if (u, v) in undirected_edges:
                continue
            
            undirected_edges.add((u, v))

        # create the final list of directed edges from the undirected pairs.
        final_edges = []
        for u, v in undirected_edges:
            final_edges.append([u, v])
            final_edges.append([v, u])

        edge_index = torch.tensor(final_edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def generate_minimum_spanning_tree(self, num_nodes):
        """
        Generate a minimum spanning tree with bidirectional edges.
        This creates a graph with the minimum number of edges needed for connectivity.
        """
        if num_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long)
        
        # Create a random spanning tree
        nodes = list(range(num_nodes))
        random.shuffle(nodes)
        undirected_edges = set()
        
        # We only need (num_nodes - 1) edges for a spanning tree
        for i in range(1, num_nodes):
            src = nodes[i]
            tgt = random.choice(nodes[:i])
            u, v = min(src, tgt), max(src, tgt)
            undirected_edges.add((u, v))
        
        # Create bidirectional edges
        final_edges = []
        for u, v in undirected_edges:
            final_edges.append([u, v])
            final_edges.append([v, u])
        
        edge_index = torch.tensor(final_edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    @staticmethod
    def convert_embedding(embeddings_df):
        def safe_convert(embedding):
            if isinstance(embedding, str):
                try:
                    embedding = ast.literal_eval(embedding)
                except (SyntaxError, ValueError):
                    try:
                        if embedding.startswith('[') and embedding.endswith(']'):
                            values = embedding[1:-1].split()
                            embedding = np.array([float(v) for v in values])
                            return embedding
                    except Exception as e:
                        return None
            if isinstance(embedding, list):
                try:
                    return np.array([float(x) for x in embedding])
                except ValueError:
                    return None
            if isinstance(embedding, np.ndarray):
                try:
                    return embedding.astype(float)
                except ValueError:
                    return None
            return None

        embeddings_df['embedding'] = embeddings_df['embedding'].apply(safe_convert)
        return embeddings_df

    @property
    def num_edge_features(self):
        """Return the number of edge features based on configuration."""
        if self.use_contact_embeddings:
            return self.contact_embedding_dim
        elif self.use_scalar_contacts:
            return 1
        return 1

    def _validate_embedding(self, emb):
        """Returns the embedding if its length matches the global dimension, otherwise None."""
        if emb is None or len(emb) != self.contact_embedding_dim:
            return None
        return emb

    def _find_contact_embedding(self, contact_embeddings, contact_id):
        """Finds a contact embedding using multiple key formats."""
        # Strategy 1: Try with 'contact_' prefix (e.g., 'contact_0_8')
        prefixed_key = f"contact_{contact_id}"
        if prefixed_key in contact_embeddings:
            return contact_embeddings[prefixed_key]

        # Strategy 2: Try the ID directly (e.g., '0_8')
        if contact_id in contact_embeddings:
            return contact_embeddings[contact_id]
            
        # Strategy 3: Try with 'contacts_' (plural) prefix
        plural_key = f"contacts_{contact_id}"
        if plural_key in contact_embeddings:
            return contact_embeddings[plural_key]
            
        return None

    def process_data(self):
        # Setup logging for skipped assemblies
        log_path = os.path.join(self.processed_dir, 'skipped_assemblies.log')
        logging.info("Logging skipped assemblies to: %s", log_path)
        with open(log_path, 'w') as f:
            f.write("assembly_id,reason,details\n")

        def log_skip(assembly_id, reason, details=""):
            """Helper to log a skipped assembly to the file and console."""
            with open(log_path, 'a') as f:
                f.write(f'"{assembly_id}","{reason}","{details}"\n')
            
            # Update summary dictionary
            if reason not in skip_reasons:
                skip_reasons[reason] = []
            skip_reasons[reason].append(assembly_id)

        valid_assemblies = pd.read_parquet(self.valid_assemblies_path)
        embeddings = pd.read_parquet(self.embeddings_path)
        if 'filename' in embeddings.columns and 'uuid' not in embeddings.columns:
            embeddings.rename(columns={'filename': 'uuid'}, inplace=True)
        if 'assembly_ID' in embeddings.columns and 'assembly_id' not in embeddings.columns:
            embeddings.rename(columns={'assembly_ID': 'assembly_id'}, inplace=True)
        embeddings = AssemblyGraphDataset.convert_embedding(embeddings)
        
        uuid_to_embedding = dict(zip(embeddings['uuid'], embeddings['embedding']))
        
        edge_attributes = None
        if self.use_scalar_contacts:
            edge_attributes = AssemblyGraph.load_edge_attributes_from_parquet(self.contacts_path)

        logging.info("=== DEBUG INFO ===")
        logging.info("Valid assemblies count: %d", len(valid_assemblies))
        logging.info("Embeddings count: %d", len(embeddings))
        if edge_attributes:
            logging.info("Edge attributes count: %d", len(edge_attributes))
        logging.info("Contact embeddings available: %s", len(self.contact_embeddings_map) > 0)
        logging.info("==================")

        skip_reasons = {}
        data_list = []
        
        # debug counters
        total_edges_processed = 0
        successful_contact_matches = 0
        
        # enhanced debug counters
        total_assemblies_debugged = 0
        edge_mismatch_details = {
            'bidirectional_duplicates': 0,
            'missing_contact_ids': 0,
            'contact_id_format_mismatches': 0,
            'embedding_not_found': 0,
            'other_failures': 0
        }
        
        for assembly_id in tqdm(valid_assemblies['assembly_id'], desc="Processing Assemblies"):
            assembly_dir = os.path.join(self.root, assembly_id)
            assembly_json_path = os.path.join(assembly_dir, 'assembly.json')
            
            if not os.path.isfile(assembly_json_path):
                log_skip(assembly_id, 'missing_json', f"File not found: {assembly_json_path}")
                continue
                
            ag = AssemblyGraph(assembly_json_path)
            ag.get_graph_data(edge_attributes=edge_attributes)
            node_ids = [node['id'] for node in ag.graph_nodes]
            node_features = []
            
            missing_node_id = None
            for node_id in node_ids:
                body_uuid = node_id.split("_")[-1]
                embdng = uuid_to_embedding.get(body_uuid)
                if embdng is None:
                    missing_node_id = node_id
                    break
                node_features.append(embdng)
                
            if missing_node_id:
                log_skip(assembly_id, 'missing_embeddings', f"No embedding for node: {missing_node_id}")
                continue

            # filter out isolated nodes
            connected_nodes = set()
            for link in ag.graph_links:
                connected_nodes.add(link['source'])
                connected_nodes.add(link['target'])

            filtered = [(i, nid, feat) for i, (nid, feat) in enumerate(zip(node_ids, node_features)) if nid in connected_nodes]
            if not filtered:
                log_skip(assembly_id, 'no_connected_nodes', "No nodes remained after filtering for graph connectivity.")
                continue
            
            new_indices, node_ids, node_features = zip(*filtered)
            node_ids = list(node_ids)
            node_features = list(node_features)

            id_to_new_idx = {nid: i for i, nid in enumerate(node_ids)}
            contact_embeddings = self.load_contact_embeddings(assembly_id)

            edge_index = []
            edge_attr_list = []
            
            is_debug_assembly = total_assemblies_debugged < 3
            if is_debug_assembly:
                logging.info("\n=== DETAILED DEBUG FOR ASSEMBLY %s ===", assembly_id)
                logging.info("Total graph links: %d", len(ag.graph_links))
                logging.info("Contacts in JSON: %d", len(ag.assembly_data.get('contacts', [])))
                logging.info("Available contact embeddings: %d", len(contact_embeddings))
                if len(contact_embeddings) > 0:
                    logging.info("Sample contact embedding keys: %s", list(contact_embeddings.keys())[:5])
                contacts_in_json = ag.assembly_data.get('contacts', [])
                if len(contacts_in_json) > 0:
                    logging.info("Sample contact IDs from JSON: %s", [c.get('id', 'NO_ID') for c in contacts_in_json[:5]])
            
            unmatched_edges = []
            matched_edges = []
            
            for link in ag.graph_links:
                src = link['source']
                tgt = link['target']
                if src in id_to_new_idx and tgt in id_to_new_idx:
                    src_idx = id_to_new_idx[src]
                    tgt_idx = id_to_new_idx[tgt]
                    edge_index.append([src_idx, tgt_idx])
                    edge_features = []
                    if self.use_scalar_contacts:
                        contact_label = link.get('contact_label', 0.0)
                        edge_features.append(contact_label)
                    if self.use_contact_embeddings:
                        total_edges_processed += 1
                        assembly_contact_id = link.get('contact_id')
                        embedding = None
                        if assembly_contact_id:
                            embedding = self._find_contact_embedding(contact_embeddings, assembly_contact_id)
                            if embedding is not None:
                                successful_contact_matches += 1
                            else:
                                edge_mismatch_details['embedding_not_found'] += 1
                        else:
                            edge_mismatch_details['missing_contact_ids'] += 1
                        embedding = self._validate_embedding(embedding)
                        if embedding is not None:
                            norm = np.linalg.norm(embedding)
                            if norm > 1e-6:  # avoid division by zero
                                embedding = embedding / norm
                            edge_features.extend(embedding.tolist())
                        else:
                            zero_embedding = [0.0] * self.contact_embedding_dim
                            edge_features.extend(zero_embedding)
                            if is_debug_assembly:
                                if not assembly_contact_id:
                                    logging.warning("Using zero embedding for edge %s->%s because original contact in JSON is missing an 'id' field.", src, tgt)
                                else:
                                    logging.warning("Using zero embedding for edge %s->%s, could not find embedding for contact_id: %s", src, tgt, assembly_contact_id)
                    if not edge_features:
                        edge_features = [1.0]
                    edge_attr_list.append(edge_features)
            
            if is_debug_assembly:
                logging.info("Matched edges: %d", len(matched_edges))
                logging.info("Unmatched edges: %d", len(unmatched_edges))
                if len(unmatched_edges) > 0:
                    logging.info("Sample unmatched edges: %s", unmatched_edges[:3])
                    no_contact_id = sum(1 for _, _, cid, _, _ in unmatched_edges if cid is None or cid == '')
                    has_contact_id_no_embedding = sum(1 for _, _, cid, _, found in unmatched_edges if cid and not found)
                    
                    logging.info("  No contact ID found: %d", no_contact_id)
                    logging.info("  Contact ID found but no embedding: %d", has_contact_id_no_embedding)
                    
                total_assemblies_debugged += 1
            
            if not edge_index:
                log_skip(assembly_id, 'no_connected_nodes', "No valid edges after remapping.")
                continue
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

            num_nodes = x.shape[0]
            masked_node_idx = random.randint(0, num_nodes-1)
            y = x[masked_node_idx].clone().unsqueeze(0)
            x[masked_node_idx] = 0

            if y.shape != torch.Size([1, 256]):
                log_skip(assembly_id, 'invalid_y_shape', f"Target y had incorrect shape: {y.shape}")
                continue

            data = Data(
                x=x, 
                edge_index=edge_index, 
                edge_attr=edge_attr, 
                y=y,
                assembly_id=assembly_id,
                masked_node_idx=masked_node_idx
            )
            data.node_uuids = node_ids
            data_list.append(data)
            
        logging.info("\n=== PROCESSING SUMMARY ===")
        logging.info("Total input assemblies: %d", len(valid_assemblies))
        logging.info("Successfully processed: %d", len(data_list))
        logging.info("Total skipped: %d", sum(len(ids) for ids in skip_reasons.values()))
        logging.info("Contact embedding stats:")
        logging.info("  Total edges processed for embeddings: %d", total_edges_processed)
        logging.info("  Embeddings successfully found: %d", successful_contact_matches)
        match_rate = (successful_contact_matches / total_edges_processed * 100) if total_edges_processed > 0 else 0
        logging.info("  Match rate: %.1f%%", match_rate)
        
        logging.info("\n=== EDGE MISMATCH ANALYSIS ===")
        logging.info("  Edges from contacts missing 'id' in JSON: %d", edge_mismatch_details.get('missing_contact_ids', 0))
        logging.info("  Embeddings not found for valid contact ID: %d", edge_mismatch_details.get('embedding_not_found', 0))
        logging.info("=========================")

        if data_list:
            sample_edge_attr_dim = data_list[0].edge_attr.shape[1]
            logging.info("Edge attribute dimension: %d", sample_edge_attr_dim)
        
        logging.info("\n=== EDGE MISMATCH ANALYSIS ===")
        logging.info("Mismatch breakdown:")
        for reason, count in edge_mismatch_details.items():
            logging.info("  %s: %d", reason, count)
        logging.info("=========================")
        
        if not data_list:
            raise ValueError("No valid graphs were processed. Please check the logs for reasons why assemblies were skipped. Common reasons include missing node embeddings or lack of connected nodes in the assembly graphs.")
            
        logging.info("Data processed and saved at %s", self.processed_paths[0])
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save(self.collate(data_list), self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    
    def process_data_no_edge_attr(self):
        logging.info("Processing data without edge attributes...")
        valid_assemblies = pd.read_parquet(self.valid_assemblies_path)
        embeddings = pd.read_parquet(self.embeddings_path)
        if 'filename' in embeddings.columns and 'uuid' not in embeddings.columns:
            embeddings.rename(columns={'filename': 'uuid'}, inplace=True)
        if 'assembly_ID' in embeddings.columns and 'assembly_id' not in embeddings.columns:
            embeddings.rename(columns={'assembly_ID': 'assembly_id'}, inplace=True)
        embeddings = AssemblyGraphDataset.convert_embedding(embeddings)

        contacts = pd.read_parquet(self.contacts_path)
        uuid_to_embedding = dict(zip(embeddings['uuid'], embeddings['embedding']))
        data_list = []
        skipped_assemblies = 0

        for assembly_id in tqdm(valid_assemblies['assembly_id'], desc="Processing Assemblies"):
            assembly_dir = os.path.join(self.root, assembly_id)
            assembly_json_path = os.path.join(assembly_dir, 'assembly.json')
            if not os.path.isfile(assembly_json_path):
                logging.warning("Skipping assembly %s: Missing assembly.json file.", assembly_id)
                skipped_assemblies += 1
                continue

            ag = AssemblyGraph(assembly_json_path)
            ag.get_graph_data()
            node_ids = [node['id'] for node in ag.graph_nodes]
            node_features = []
            missing = False
            for node_id in node_ids:
                body_uuid = node_id.split("_")[-1]
                embdng = uuid_to_embedding.get(body_uuid)
                if embdng is None:
                    logging.warning("Skipping assembly %s: Missing node embeddings.", assembly_id)
                    missing = True
                    break
                node_features.append(embdng)
            if missing:
                skipped_assemblies += 1
                continue

            # filter out isolated nodes
            connected_nodes = set()
            for link in ag.graph_links:
                connected_nodes.add(link['source'])
                connected_nodes.add(link['target'])

            # keep nodes that are connected
            filtered = [(i, nid, feat) for i, (nid, feat) in enumerate(zip(node_ids, node_features)) if nid in connected_nodes]
            if not filtered:
                skipped_assemblies += 1
                continue

            # unpack filtered nodes
            new_indices, node_ids, node_features = zip(*filtered)
            node_ids = list(node_ids)
            node_features = list(node_features)

            # map node id to new index
            id_to_new_idx = {nid: i for i, nid in enumerate(node_ids)}

            x = torch.tensor(node_features, dtype=torch.float)

            # filter and remap edges
            edge_index = []
            for link in ag.graph_links:
                src = link['source']
                tgt = link['target']
                if src in id_to_new_idx and tgt in id_to_new_idx:
                    edge_index.append([id_to_new_idx[src], id_to_new_idx[tgt]])

            if not edge_index:
                logging.warning("Skipping assembly %s: Empty edge_index.", assembly_id)
                skipped_assemblies += 1
                continue
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            num_nodes = x.size(0)
            masked_node_idx = random.randint(0, num_nodes - 1)
            y = x[masked_node_idx].clone().unsqueeze(0)
            x[masked_node_idx] = 0

            if y.shape != torch.Size([1, 256]):
                logging.error("Error: Target y has incorrect shape: %s. Skipping assembly.", y.shape)
                skipped_assemblies += 1
                continue

            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                assembly_id=assembly_id,
                masked_node_idx=masked_node_idx
            )
            data.node_uuids = node_ids
            if data.x is not None and data.edge_index is not None and data.y is not None:
                data_list.append(data)
            else:
                logging.warning("Skipping assembly %s: Invalid Data object.", assembly_id)
                skipped_assemblies += 1

        if len(data_list) == 0:
            raise ValueError("No valid graphs were processed. Check your data processing logic.")

        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save(self.collate(data_list), self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        logging.info("Data processed and saved at %s", self.processed_paths[0])
        logging.info("Total assemblies skipped: %d", skipped_assemblies)
        logging.info("Total valid graphs processed: %d", len(data_list))

    

    def process_data_classification(self):
        logging.info("Processing data for classification with edge attributes...")
        valid_assemblies = pd.read_parquet(self.valid_assemblies_path)
        embeddings = pd.read_parquet(self.embeddings_path)
        if 'filename' in embeddings.columns and 'uuid' not in embeddings.columns:
            embeddings.rename(columns={'filename': 'uuid'}, inplace=True)
        if 'assembly_ID' in embeddings.columns and 'assembly_id' not in embeddings.columns:
            embeddings.rename(columns={'assembly_ID': 'assembly_id'}, inplace=True)
        embeddings = AssemblyGraphDataset.convert_embedding(embeddings)
        
        uuid_to_embedding = dict(zip(embeddings['uuid'], embeddings['embedding']))
        edge_attributes = None
        if self.use_scalar_contacts:
            edge_attributes = AssemblyGraph.load_edge_attributes_from_parquet(self.contacts_path)
        precomputed_clusters_path = os.path.join(self.processed_dir, f'precomputed_clusters_{self.num_clusters}_{self.embedding_suffix}.parquet')
        logging.info("precomputed_clusters_path: %s", precomputed_clusters_path)

        if os.path.isfile(precomputed_clusters_path):
            logging.info("Loading precomputed clusters from %s...", precomputed_clusters_path)
            precomputed_clusters = pd.read_parquet(precomputed_clusters_path)
            uuid_to_cluster_label = dict(zip(precomputed_clusters['uuid'], precomputed_clusters['cluster_label']))
        else:
            logging.info("Precomputed clusters not found. Performing clustering...")
            all_embeddings = []
            valid_uuids = []
            for assembly_id in valid_assemblies['assembly_id']:
                assembly_dir = os.path.join(self.root, assembly_id)
                assembly_json_path = os.path.join(assembly_dir, 'assembly.json')
                if not os.path.isfile(assembly_json_path):
                    continue
                ag = AssemblyGraph(assembly_json_path)
                ag.get_graph_data()
                node_ids = [node['id'] for node in ag.graph_nodes]
                for node_id in node_ids:
                    body_uuid = node_id.split("_")[-1]
                    embdng = uuid_to_embedding.get(body_uuid)
                    if embdng is not None:
                        all_embeddings.append(embdng)
                        valid_uuids.append(body_uuid)

            # clustering
            logging.info("Clustering node embeddings...")
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(all_embeddings)
            uuid_to_cluster_label = dict(zip(valid_uuids, cluster_labels))
            logging.info("Saving precomputed clusters to %s...", precomputed_clusters_path)
            precomputed_clusters = pd.DataFrame({'uuid': valid_uuids, 'cluster_label': cluster_labels})
            os.makedirs(os.path.dirname(precomputed_clusters_path), exist_ok=True)
            precomputed_clusters.to_parquet(precomputed_clusters_path)
            kmeans_model_path = os.path.join(self.processed_dir, f'kmeans_model_{self.num_clusters}_{self.embedding_suffix}.pkl')
            logging.info("Saving KMeans model to %s...", kmeans_model_path)
            joblib.dump(kmeans, kmeans_model_path)

        data_list = []
        skipped_assemblies = 0

        # process assemblies
        for assembly_id in tqdm(valid_assemblies['assembly_id'], desc="Processing Assemblies"):
            assembly_dir = os.path.join(self.root, assembly_id)
            assembly_json_path = os.path.join(assembly_dir, 'assembly.json')
            if not os.path.isfile(assembly_json_path):
                skipped_assemblies += 1
                continue
            ag = AssemblyGraph(assembly_json_path)
            ag.get_graph_data(edge_attributes=edge_attributes)
            node_ids = [node['id'] for node in ag.graph_nodes]
            node_features = []
            node_labels = []
            missing = False
            for node_id in node_ids:
                body_uuid = node_id.split("_")[-1]
                embdng = uuid_to_embedding.get(body_uuid)
                cluster_label = uuid_to_cluster_label.get(body_uuid)
                if embdng is None or cluster_label is None:
                    missing = True
                    break
                node_features.append(embdng)
                node_labels.append(cluster_label)
            if missing:
                skipped_assemblies += 1
                continue

            # filter out isolated nodes
            connected_nodes = set()
            for link in ag.graph_links:
                connected_nodes.add(link['source'])
                connected_nodes.add(link['target'])

            # only keep nodes that are connected
            filtered = [(i, nid, feat, label) for i, (nid, feat, label) in enumerate(zip(node_ids, node_features, node_labels)) if nid in connected_nodes]
            if not filtered:
                skipped_assemblies += 1
                continue

            # unpack filtered nodes
            new_indices, node_ids, node_features, node_labels = zip(*filtered)
            node_ids = list(node_ids)
            node_features = list(node_features)
            node_labels = list(node_labels)

            # map node id to new index
            id_to_new_idx = {nid: i for i, nid in enumerate(node_ids)}

            x = torch.tensor(node_features, dtype=torch.float)
            
            # masked node prediction setup
            num_nodes = x.size(0)
            if num_nodes == 0:
                skipped_assemblies += 1
                continue

            masked_node_idx = random.randint(0, num_nodes - 1)
            y_cls = torch.tensor([node_labels[masked_node_idx]], dtype=torch.long) # target is the label of the masked node
            x[masked_node_idx] = 0

            # filter and remap edges and edge_attr
            edge_index = []
            edge_attr = []
            for link in ag.graph_links:
                src = link['source']
                tgt = link['target']
                if src in id_to_new_idx and tgt in id_to_new_idx:
                    edge_index.append([id_to_new_idx[src], id_to_new_idx[tgt]])
                    edge_features = []
                    if self.use_scalar_contacts and not self.use_contact_embeddings:
                        contact_label = link.get('contact_label')
                        if contact_label is None:
                            contact_label = 0.0
                        edge_features.append(contact_label)
                    if self.use_contact_embeddings:
                        contact_embeddings = self.load_contact_embeddings(assembly_id)
                        contact_id = link.get('contact_id') or link.get('id')
                        emb = self._find_contact_embedding(contact_embeddings, contact_id) if contact_id else None
                        emb = self._validate_embedding(emb)
                        if emb is not None:
                            norm = np.linalg.norm(emb)
                            if norm > 1e-6:
                                emb = emb / norm
                        else:
                            emb = np.zeros(self.contact_embedding_dim, dtype=np.float32)
                        edge_features.extend(emb.tolist())
                    edge_attr.append(edge_features)
            if not edge_index:
                skipped_assemblies += 1
                continue
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y_cls=y_cls,
                assembly_id=assembly_id,
                masked_node_idx=torch.tensor([masked_node_idx], dtype=torch.long)
            )
            data.node_uuids = node_ids
            data_list.append(data)
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save(self.collate(data_list), self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        logging.info("Data processed and saved at %s", self.processed_paths[0])
        logging.info("Total assemblies skipped: %d", skipped_assemblies)


    def process_data_classification_no_edge_attr(self):
        logging.info("Processing data for classification without edge attributes...")
        valid_assemblies = pd.read_parquet(self.valid_assemblies_path)
        embeddings = pd.read_parquet(self.embeddings_path)
        if 'filename' in embeddings.columns and 'uuid' not in embeddings.columns:
            embeddings.rename(columns={'filename': 'uuid'}, inplace=True)
        if 'assembly_ID' in embeddings.columns and 'assembly_id' not in embeddings.columns:
            embeddings.rename(columns={'assembly_ID': 'assembly_id'}, inplace=True)
        embeddings = AssemblyGraphDataset.convert_embedding(embeddings)
        uuid_to_embedding = dict(zip(embeddings['uuid'], embeddings['embedding']))
        precomputed_clusters_path = os.path.join(self.processed_dir, f'precomputed_clusters_{self.num_clusters}_{self.embedding_suffix}.parquet')
        logging.info("precomputed_clusters_path: %s", precomputed_clusters_path)

        if os.path.isfile(precomputed_clusters_path):
            logging.info("Loading precomputed clusters from %s...", precomputed_clusters_path)
            precomputed_clusters = pd.read_parquet(precomputed_clusters_path)
            uuid_to_cluster_label = dict(zip(precomputed_clusters['uuid'], precomputed_clusters['cluster_label']))
        else:
            logging.info("Precomputed clusters not found. Performing clustering...")
            all_embeddings = []
            valid_uuids = []
            for assembly_id in valid_assemblies['assembly_id']:
                assembly_dir = os.path.join(self.root, assembly_id)
                assembly_json_path = os.path.join(assembly_dir, 'assembly.json')
                if not os.path.isfile(assembly_json_path):
                    continue
                ag = AssemblyGraph(assembly_json_path)
                ag.get_graph_data()
                node_ids = [node['id'] for node in ag.graph_nodes]
                for node_id in node_ids:
                    body_uuid = node_id.split("_")[-1]
                    embdng = uuid_to_embedding.get(body_uuid)
                    if embdng is not None:
                        all_embeddings.append(embdng)
                        valid_uuids.append(body_uuid)

            # clustering
            logging.info("Clustering node embeddings...")
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(all_embeddings)
            uuid_to_cluster_label = dict(zip(valid_uuids, cluster_labels))
            logging.info("Saving precomputed clusters to %s...", precomputed_clusters_path)
            precomputed_clusters = pd.DataFrame({'uuid': valid_uuids, 'cluster_label': cluster_labels})
            os.makedirs(os.path.dirname(precomputed_clusters_path), exist_ok=True)
            precomputed_clusters.to_parquet(precomputed_clusters_path)
            kmeans_model_path = os.path.join(self.processed_dir, f'kmeans_model_{self.num_clusters}_{self.embedding_suffix}.pkl')
            logging.info("Saving KMeans model to %s...", kmeans_model_path)
            joblib.dump(kmeans, kmeans_model_path)

        data_list = []
        skipped_assemblies = 0

        # process assemblies
        for assembly_id in tqdm(valid_assemblies['assembly_id'], desc="Processing Assemblies"):
            assembly_dir = os.path.join(self.root, assembly_id)
            assembly_json_path = os.path.join(assembly_dir, 'assembly.json')
            if not os.path.isfile(assembly_json_path):
                skipped_assemblies += 1
                continue
            ag = AssemblyGraph(assembly_json_path)
            ag.get_graph_data()
            node_ids = [node['id'] for node in ag.graph_nodes]
            node_features = []
            node_labels = []
            missing = False
            for node_id in node_ids:
                body_uuid = node_id.split("_")[-1]
                embdng = uuid_to_embedding.get(body_uuid)
                cluster_label = uuid_to_cluster_label.get(body_uuid)
                if embdng is None or cluster_label is None:
                    missing = True
                    break
                node_features.append(embdng)
                node_labels.append(cluster_label)
            if missing:
                skipped_assemblies += 1
                continue

            # filter out isolated nodes
            connected_nodes = set()
            for link in ag.graph_links:
                connected_nodes.add(link['source'])
                connected_nodes.add(link['target'])

            # only keep nodes that are connected
            filtered = [(i, nid, feat, label) for i, (nid, feat, label) in enumerate(zip(node_ids, node_features, node_labels)) if nid in connected_nodes]
            if not filtered:
                skipped_assemblies += 1
                continue

            # unpack filtered nodes
            new_indices, node_ids, node_features, node_labels = zip(*filtered)
            node_ids = list(node_ids)
            node_features = list(node_features)
            node_labels = list(node_labels)

            # map node id to new index
            id_to_new_idx = {nid: i for i, nid in enumerate(node_ids)}

            x = torch.tensor(node_features, dtype=torch.float)
            
            # masked node prediction setup
            num_nodes = x.size(0)
            if num_nodes == 0:
                skipped_assemblies += 1
                continue

            masked_node_idx = random.randint(0, num_nodes - 1)
            y_cls = torch.tensor([node_labels[masked_node_idx]], dtype=torch.long) # target is the label of the masked node
            x[masked_node_idx] = 0

            # filter and remap edges
            edge_index = []
            for link in ag.graph_links:
                src = link['source']
                tgt = link['target']
                if src in id_to_new_idx and tgt in id_to_new_idx:
                    edge_index.append([id_to_new_idx[src], id_to_new_idx[tgt]])
            if not edge_index:
                skipped_assemblies += 1
                continue
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            data = Data(
                x=x,
                edge_index=edge_index,
                y_cls=y_cls,
                assembly_id=assembly_id,
                masked_node_idx=torch.tensor([masked_node_idx], dtype=torch.long)
            )
            data.node_uuids = node_ids
            data_list.append(data)

        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save(self.collate(data_list), self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        logging.info("Data processed and saved at %s", self.processed_paths[0])
        logging.info("Total assemblies skipped: %d", skipped_assemblies)

    def process_data_augmentation(
        self,
        include_partial: bool = True,
        include_full: bool = True,
        include_random: bool = True,
        include_mst: bool = True,
    ):
        """
        Build three augmented datasets:

            _RE.pt      : random-edge graphs
            _MST.pt     : minimum spanning tree graphs
            _PARCON.pt  : partial graphs (edge-drop)
            _FULCON.pt  : fully-connected graphs

        Each augmented graph shares x / y / node_uuids with the base graph,
        so no large tensors are duplicated.
        """
        original_path = self.processed_paths[0]
        self.data, self.slices = torch.load(original_path, weights_only=False)

        # returns new Data object that re-uses heavy tensors
        def _clone_shallow(src_graph):
            return Data(
                x               = src_graph.x,
                y               = getattr(src_graph, "y",     None),
                y_cls           = getattr(src_graph, "y_cls", None),
                assembly_id     = src_graph.assembly_id,
                masked_node_idx = getattr(src_graph, "masked_node_idx", None),
                node_uuids      = src_graph.node_uuids,
            )

        re_list, pc_list, fc_list, mst_list = [], [], [], []

        for g in self:
            n_nodes = g.x.size(0)
            n_edges = g.edge_index.size(1)

            # random-edge variant
            if include_random:
                r  = _clone_shallow(g)
                r.edge_index = self.generate_random_edges(n_nodes, n_edges)
                r.edge_attr = None  # Explicitly tell PyG there are no edge attributes
                r.aug_type = "random_edges"
                re_list.append(r)

            # minimum Spanning Tree variant
            if include_mst:
                m = _clone_shallow(g)
                m.edge_index = self.generate_minimum_spanning_tree(n_nodes)
                m.edge_attr = None
                m.aug_type = "minimum_spanning_tree"
                mst_list.append(m)

            # partial-connection variant
            if include_partial:
                p  = _clone_shallow(g)
                keep = torch.randperm(n_edges)[: int(n_edges * (1 - self.aug_fraction))]
                p.edge_index = g.edge_index[:, keep]
                if hasattr(g, "edge_attr") and g.edge_attr is not None:
                    p.edge_attr = g.edge_attr[keep]
                p.aug_type = "partial_drop"
                pc_list.append(p)

            # fully-connected variant
            if include_full:
                f  = _clone_shallow(g)
                row, col = zip(*[(i, j) for i in range(n_nodes)
                                          for j in range(n_nodes) if i != j])
                f.edge_index = torch.tensor([row, col], dtype=torch.long)
                f.aug_type = "fully_connected"
                fc_list.append(f)

        # helper for destination names
        def _dest(tag): return original_path.replace(".pt", f"_{tag}.pt")

        if include_random and re_list:
            torch.save(self.collate(re_list), _dest("RE"))
        if include_mst and mst_list:
            torch.save(self.collate(mst_list), _dest("MST"))
        if include_partial and pc_list:
            parcon_tag = f"PARCON_{int(self.aug_fraction * 100)}"
            torch.save(self.collate(pc_list), _dest(parcon_tag))
        if include_full and fc_list:
            torch.save(self.collate(fc_list), _dest("FULCON"))

        logging.info("[Aug]  saved: %d random-edge | %d mst | %d partial-drop | %d fully-connected",
              len(re_list),
              len(mst_list),
              len(pc_list),
              len(fc_list))

        # restore base graphs in memory
        self.data, self.slices = torch.load(original_path, weights_only=False)

    @property
    def processed_file_names(self):
        suffix = ""
        if self.use_scalar_contacts and self.use_contact_embeddings:
            suffix = f"_scalar_and_embeddings_{self.contact_embedding_dim}"
        elif self.use_scalar_contacts:
            suffix = "_scalar_only"
        elif self.use_contact_embeddings:
            suffix = f"_embeddings_only_{self.contact_embedding_dim}"
        else:
            suffix = "_no_contacts"
        
        # add edge feature mode to the filename to avoid cache collisions
        suffix += f"_mode_{self.edge_feature_mode}"

        filename = None
        if self.model_type == 'GATv2':
            filename = f'assembly_graph_data_new_contact{suffix}.pt'
        elif self.model_type == 'GATv2NoEdgeAttr':
            filename = f'assembly_graph_data_zero_edge_attr_new_contact{suffix}.pt'
        elif self.model_type == 'GATv2Classification' or self.model_type == 'GATClassification':
            suffix += f"_clusters_{self.num_clusters}"
            suffix += f"_embeds_{self.embedding_suffix}"
            filename = f'assembly_graph_data_classification_new_contact{suffix}.pt'
        elif self.model_type == 'GATv2ClassificationNoEdgeAttr':
            suffix += f"_clusters_{self.num_clusters}"
            suffix += f"_embeds_{self.embedding_suffix}"
            filename = f'assembly_graph_data_classification_no_edge_attr_new_contact{suffix}.pt'
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        if self.data_id:
            base, ext = os.path.splitext(filename)
            filename = f"{base}_id_{self.data_id}{ext}"
        
        return [filename]

    def get(self, idx):
        data = super().get(idx)
        if isinstance(data.assembly_id, list):
            data.assembly_id = data.assembly_id[0]
        return data