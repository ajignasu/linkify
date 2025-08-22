import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path


class AssemblyGraph():
    """
    Construct a graph representing an assembly with connectivity
    between visible B-Rep bodies with joints and contact surfaces
    """

    def __init__(self, assembly_data):
        if isinstance(assembly_data, dict):
            self.assembly_data = assembly_data
        else:
            if isinstance(assembly_data, str):
                assembly_file = Path(assembly_data)
            else:
                assembly_file = assembly_data
            assert assembly_file.exists()
            with open(assembly_file, "r", encoding="utf-8") as f:
                self.assembly_data = json.load(f)
        self.graph_nodes = []
        self.graph_links = []
        self.graph_node_ids = set()
    
    @staticmethod
    def load_node_attributes_from_parquet(parquet_path, assembly_id=None, id_column="None"):
        df = pd.read_parquet(parquet_path)
        if assembly_id:
            df = df[df['assembly_id'] == assembly_id]

        # this probably will change once i get the part embeddings. For now, its just for debugging.
        if id_column not in df.columns:
            raise KeyError(f"Parquet file must have '{id_column}' column.")
        return df.set_index(id_column).to_dict(orient='index')

    @staticmethod
    def load_edge_attributes_from_parquet(parquet_path, assembly_id="None"):
        df = pd.read_parquet(parquet_path)
        edge_attributes = {}
        # Ensure 'link_id' is the index for efficient lookup
        if 'link_id' in df.columns:
            df.set_index('link_id', inplace=True)
        
        # Convert the DataFrame to a dictionary of dictionaries
        # Format: {link_id: {col1: val1, col2: val2, ...}}
        edge_attributes = df.to_dict(orient='index')
        return edge_attributes

    def get_graph_data(self, node_attributes=None, edge_attributes=None):
        """Get the graph data as a list of nodes and links"""
        self.graph_nodes = []
        self.graph_links = []
        self.graph_node_ids = set()
        # TODO: Add support for a flag to include hidden bodies
        self.populate_graph_nodes(node_attributes)
        self.populate_graph_links(edge_attributes)
        if node_attributes:
            for node in self.graph_nodes:
                attrs = node_attributes.get(node["id"], {})
                if attrs:
                    node.update(attrs)
        if edge_attributes:
            for link in self.graph_links:
                link_id = link["id"]
                reverse_link_id = ">".join(link_id.split(">")[::-1])
                
                # Retrieve all attributes for the given link_id
                attributes = edge_attributes.get(link_id, None)
                if attributes is None:
                    attributes = edge_attributes.get(reverse_link_id, None)
                
                if attributes is not None:
                    # Update the link with all attributes from the parquet file
                    link.update(attributes)

        return self.graph_nodes, self.graph_links

    def get_graph_networkx(self):
        """Get a networkx graph"""
        graph_data = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [],
            "links": []
        }
        graph_data["nodes"], graph_data["links"] = self.get_graph_data()
        from networkx.readwrite import json_graph
        return json_graph.node_link_graph(graph_data)

    def get_node_label_dict(self, attribute="occurrence_name"):
        """Get a dictionary mapping from node ids to a given attribute"""
        label_dict = {}
        if len(self.graph_nodes) == 0:
            return label_dict
        for node in self.graph_nodes:
            node_id = node["id"]
            if attribute in node:
                node_att = node[attribute]
            else:
                node_att = node["body_name"]
            label_dict[node_id] = node_att
        return label_dict

    def export_graph_json(self, json_file, node_attributes, edge_attributes, include_attributes=False):
        """Export the graph as an networkx node-link format json file"""
        graph_data = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [],
            "links": []
        }
        graph_data["nodes"], graph_data["links"] = self.get_graph_data(node_attributes, edge_attributes)

        # print("graph_data: ", graph_data)

        if include_attributes:
            for node in graph_data["nodes"]:
                node["embedding"] = node.get("embedding", None)
            
            for link in graph_data["links"]:
                # print("link[contact_label]: ", link.get("contact_label", None))
                link["contact_label"] = link.get("contact_label", None)
        

        # print("\ngraph_data after adding attributes: ", graph_data)


        with open(json_file, "w", encoding="utf8") as f:
            json.dump(graph_data, f, indent=4)
        return json_file.exists()

    def populate_graph_nodes(self, node_attributes=None):
        """
        Recursively traverse the assembly tree
        and generate a flat set of graph nodes from bodies
        """
        # First get the root and add it's bodies
        root_component_uuid = self.assembly_data["root"]["component"]
        root_component = self.assembly_data["components"][root_component_uuid]
        if "bodies" in root_component:
            for body_uuid in root_component["bodies"]:
                node_data = self.get_graph_node_data(body_uuid)
                if node_attributes and node_data["id"] in node_attributes:
                    node_data.update(node_attributes[node_data["id"]])
                self.graph_nodes.append(node_data)
        # Recurse over the occurrences in the tree
        tree_root = self.assembly_data["tree"]["root"]
        root_transform = np.identity(4)
        self.walk_tree(tree_root, root_transform)
        # Ensure all node IDs are unique
        total_nodes = len(self.graph_nodes)
        self.graph_node_ids = set([f["id"] for f in self.graph_nodes])
        assert total_nodes == len(self.graph_node_ids), "Duplicate node ids found"

    def populate_graph_links(self, contact_labels=None):
        """Create links in the graph between bodies with joints and contacts"""
        # if "joints" in self.assembly_data:
        #     self.populate_graph_joint_links()
        # if "as_built_joints" in self.assembly_data:
        #     self.populate_graph_as_built_joint_links()
        if "contacts" in self.assembly_data:
            self.populate_graph_contact_links(contact_labels)

    def walk_tree(self, occ_tree, occ_transform):
        """Recursively walk the occurrence tree"""
        for occ_uuid, occ_sub_tree in occ_tree.items():
            occ = self.assembly_data["occurrences"][occ_uuid]
            if not occ["is_visible"]:
                continue
            occ_sub_transform = occ_transform @ self.transform_to_matrix(occ["transform"])
            if "bodies" in occ:
                for occ_body_uuid, occ_body in occ["bodies"].items():
                    if not occ_body["is_visible"]:
                        continue
                    node_data = self.get_graph_node_data(
                        occ_body_uuid,
                        occ_uuid,
                        occ,
                        occ_sub_transform
                    )
                    self.graph_nodes.append(node_data)
            self.walk_tree(occ_sub_tree, occ_sub_transform)

    def get_graph_node_data(self, body_uuid, occ_uuid=None, occ=None, transform=None):
        """Add a body as a graph node"""
        body = self.assembly_data["bodies"][body_uuid]
        node_data = {}
        if occ_uuid is None:
            body_id = body_uuid
        else:
            body_id = f"{occ_uuid}_{body_uuid}"
        node_data["id"] = body_id
        node_data["body_name"] = body["name"]
        node_data["body_file"] = body_uuid
        if occ:
            node_data["occurrence_name"] = occ["name"]
        if transform is None:
            transform = np.identity(4)
        node_data["transform"] = transform.tolist()
        return node_data

    def populate_graph_joint_links(self):
        """Populate directed links between bodies with joints"""
        if self.assembly_data["joints"] is None:
            pass
        else:
            for joint_uuid, joint in self.assembly_data["joints"].items():
                try:
                    ent1 = joint["geometry_or_origin_one"]["entity_one"]
                    ent2 = joint["geometry_or_origin_two"]["entity_one"]

                    body1_visible = self.is_body_visible(ent1)
                    body2_visible = self.is_body_visible(ent2)
                    if not body1_visible or not body2_visible:
                        continue
                    link_data = self.get_graph_link_data(ent1, ent2)
                    link_data["type"] = "Joint"
                    link_data["joint_type"] = joint["joint_motion"]["joint_type"]
                    self.graph_links.append(link_data)
                except:
                    continue

    def populate_graph_as_built_joint_links(self):
        """Populate directed links between bodies with as built joints"""
        if self.assembly_data["as_built_joints"] is None:
            return
        for joint_uuid, joint in self.assembly_data["as_built_joints"].items():
            geo_ent = None
            geo_ent_id = None
            # For non rigid joint types we will get geometry
            if "joint_geometry" in joint:
                if "entity_one" in joint["joint_geometry"]:
                    geo_ent = joint["joint_geometry"]["entity_one"]
                    geo_ent_id = self.get_link_entity_id(geo_ent)

            occ1 = joint["occurrence_one"]
            occ2 = joint["occurrence_two"]
            body1 = None
            body2 = None
            if geo_ent is not None and "occurrence" in geo_ent:
                if geo_ent["occurrence"] == occ1:
                    body1 = geo_ent["body"]
                elif geo_ent["occurrence"] == occ2:
                    body2 = geo_ent["body"]

            # We only add links if there is a single body
            # in both occurrences
            # TODO: Look deeper in the tree if there is a single body
            if body1 is None:
                body1 = self.get_occurrence_body_uuid(occ1)
                if body1 is None:
                    continue
            if body2 is None:
                body2 = self.get_occurrence_body_uuid(occ2)
                if body2 is None:
                    continue
            # Don't add links when the bodies aren't visible
            body1_visible = self.is_body_visible(body_uuid=body1, occurrence_uuid=occ1)
            body2_visible = self.is_body_visible(body_uuid=body2, occurrence_uuid=occ2)
            if not body1_visible or not body2_visible:
                continue
            ent1 = f"{occ1}_{body1}"
            ent2 = f"{occ2}_{body2}"
            link_id = f"{ent1}>{ent2}"
            link_data = {}
            link_data["id"] = link_id
            link_data["source"] = ent1
            assert link_data["source"] in self.graph_node_ids, "Link source id doesn't exist in nodes"
            link_data["target"] = ent2
            assert link_data["target"] in self.graph_node_ids, "Link target id doesn't exist in nodes"
            link_data["type"] = "AsBuiltJoint"
            link_data["joint_type"] = joint["joint_motion"]["joint_type"]
            # TODO: Add more joint features
            self.graph_links.append(link_data)

    def populate_graph_contact_links(self, contact_labels=None):
        """Populate undirected links between bodies in contact"""
        contacts_list = self.assembly_data.get("contacts")
        if not contacts_list:
            return
        
        # Allow processing even if contact_labels is None or empty. Use empty dict as default.
        if contact_labels is None:
            contact_labels = {}
            
        processed_pairs = set()

        for contact in contacts_list:
            ent1 = contact["entity_one"]
            ent2 = contact["entity_two"]
            
            # Create a canonical key for the pair to avoid duplicate edges
            id1 = self.get_link_entity_id(ent1)
            id2 = self.get_link_entity_id(ent2)
            pair_key = tuple(sorted((id1, id2)))
            
            # Skip if we've already created an edge for this pair
            if pair_key in processed_pairs:
                continue
            
            original_contact_id = contact.get('id')
            
            # Try to find a valid contact label first
            link_id = self.get_link_id(ent1, ent2)
            reverse_link_id = self.get_link_id(ent2, ent1)
            
            contact_label = contact_labels.get(link_id, contact_labels.get(reverse_link_id, None))
            
            # If no label provided, default to 0.0 (allows edge creation even without scalar labels)
            if contact_label is None:
                contact_label = 0.0

            # Re-enable visibility check
            body1_visible = self.is_body_visible(ent1)
            body2_visible = self.is_body_visible(ent2)
            
            if not body1_visible or not body2_visible:
                continue
            
            # Create a single, forward edge
            link_data = self.get_graph_link_data(ent1, ent2, contact_id=original_contact_id)
            link_data["type"] = "Contact"
            link_data["contact_label"] = contact_label
            self.graph_links.append(link_data)
            
            # ALSO CREATE THE REVERSE EDGE, BUT ENSURE IT HAS THE SAME ID
            link_data_reverse = self.get_graph_link_data(ent2, ent1, contact_id=original_contact_id)
            link_data_reverse["type"] = "Contact"
            link_data_reverse["contact_label"] = contact_label
            self.graph_links.append(link_data_reverse)
            
            processed_pairs.add(pair_key)

    def get_graph_link_data(self, entity_one, entity_two, contact_id=None):
        """Get the common data for a graph link from a joint or contact"""
        link_data = {}
        link_data["id"] = self.get_link_id(entity_one, entity_two)
        link_data["contact_id"] = contact_id
        link_data["source"] = self.get_link_entity_id(entity_one)
        assert link_data["source"] in self.graph_node_ids, "Link source id doesn't exist in nodes"
        link_data["target"] = self.get_link_entity_id(entity_two)
        assert link_data["target"] in self.graph_node_ids, "Link target id doesn't exist in nodes"
        return link_data

    def get_link_id(self, entity_one, entity_two):
        """Get a unique id for a link"""
        # Get body and occurrence IDs
        body_one = entity_one.get("body")
        body_two = entity_two.get("body")
        occ_one = entity_one.get("occurrence")
        occ_two = entity_two.get("occurrence")
        
        # Create full IDs in the format occurrence_uuid_body_uuid
        id_one = f"{occ_one}_{body_one}" if occ_one else body_one
        id_two = f"{occ_two}_{body_two}" if occ_two else body_two
        
        return f"{id_one}>{id_two}"

    def get_link_entity_id(self, entity):
        """Get a unique id for one side of a link"""
        # if "occurrence" in entity:
        if "occurrence" in entity and entity["occurrence"] is not None:
            return f"{entity['occurrence']}_{entity['body']}"
        else:
            return entity["body"]

    def get_occurrence_body_uuid(self, occurrence_uuid):
        """Get the body uuid from an occurrence"""
        occ = self.assembly_data["occurrences"][occurrence_uuid]
        # We only return a body_uuid if there is one body
        if "bodies" not in occ:
            return None
        if len(occ["bodies"]) != 1:
            return None
        # Return the first key
        return next(iter(occ["bodies"]))

    def is_body_visible(self, entity=None, body_uuid=None, occurrence_uuid=None):
        """Check if a body is visible"""
        if body_uuid is None:
            body_uuid = entity["body"]
        if occurrence_uuid is None:
            # If we don't have an occurrence
            # we need to look in the root component
            # if "occurrence" not in entity:
            if "occurrence" not in entity or entity["occurrence"] is None:
                body = self.assembly_data["root"]["bodies"][body_uuid]
                return body["is_visible"]
            # First check the occurrence is visible
            occurrence_uuid = entity["occurrence"]
        occ = self.assembly_data["occurrences"][occurrence_uuid]
        if not occ["is_visible"]:
            return False
        body = occ["bodies"][body_uuid]
        return body["is_visible"]

    def transform_to_matrix(self, transform=None):
        """
        Convert a transform dict into a
        4x4 affine transformation matrix
        """
        if transform is None:
            return np.identity(4)
        x_axis = self.transform_vector_to_np(transform["x_axis"])
        y_axis = self.transform_vector_to_np(transform["y_axis"])
        z_axis = self.transform_vector_to_np(transform["z_axis"])
        translation = self.transform_vector_to_np(transform["origin"])
        translation[3] = 1.0
        return np.transpose(np.stack([x_axis, y_axis, z_axis, translation]))

    def transform_vector_to_np(self, vector):
        x = vector["x"]
        y = vector["y"]
        z = vector["z"]
        h = 0.0
        return np.array([x, y, z, h])