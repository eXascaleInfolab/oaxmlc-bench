import json
import networkx as nx
import networkx.algorithms as nxalgo
import numpy as np
from matplotlib import pyplot as plt


class Taxonomy:
    """
    Represents a hierarchical taxonomy of labels for datasets, supporting operations such as loading, parsing, and managing the taxonomy structure.

    Attributes:
        cfg (dict): Configuration dictionary.
        label_to_level (dict): Maps label names to their hierarchical levels.
        level_to_labels (dict): Maps levels to lists of label names.
        label_to_idx (dict): Maps label names to their unique indices.
        idx_to_label (dict): Maps unique indices to their corresponding label names.
        label_to_abstract (dict): Maps labels to their descriptions or textual abstracts.
        label_to_title (dict): Maps labels to their textual title.
        label_has_description (dict): Maps labels to 1 if it has a textual definition, more than just the title, otherwise 0.
        label_to_children (dict): Maps labels to their child labels.
        label_to_parents (dict): Maps labels to their parent labels.
        leaves (list): List of leaf node labels in the taxonomy.
        n_nodes (int): Number of nodes in the taxonomy.
        n_edges (int): Number of edges in the taxonomy.
        height (int): Height of the taxonomy, i.e. the maximum level.
        is_a_tree (bool): Indicates whether the taxonomy is a tree or a directed acyclic graph.
        label_remappings (dict): Store remappings done after calling 'merge_nodes()'

    Methods:
        __str__(): Returns a string representation of the taxonomy.
        all_children(label): Recursively retrieves all child nodes of a given label.
        print_stats(): Prints some statistics about the taxonomy.
        is_leaf(label): Checks whether a given label is a leaf node.
        remove_leaves(leaves): Removes specified leaf nodes from the taxonomy.
        print_graph_stats(): Prints some statistics about the taxonomy view as a graph.
        get_edge_list(): Returns the edge list of the taxonomy.
        get_subtaxonomy(new_root_label): Returns a new Taxonomy object, where the label of the new root is given. 
        _recursive_all_children(label): Helper method for recursively finding all children of a node.
    """

    def __init__(self):
        # Initialize all properties
        self.label_to_level = {}
        self.level_to_labels = {}
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.label_to_abstract = {}
        self.label_to_title = {}
        self.label_has_description = {}
        self.label_to_children = {}
        self.label_to_parents = {}
        self.leaves = []
        self.n_nodes = 0
        self.n_edges = 0
        self.height = 0
        self.width = 0
        self.is_a_tree = False
        self.label_remappings = {}


    def __str__(self):
        if self.is_a_tree:
            return f"Tree with {self.n_nodes} nodes and {self.n_edges} edges"
        else:
            return f"Semi-lattice with {self.n_nodes} nodes and {self.n_edges} edges"


    def load_taxonomy(self, cfg, verbose = True):
        print(f"> Loading taxonomy...")
        self.cfg = cfg
        # Store title, label, level and description of the labels, add a top 'root' node
        self.label_to_level['root'] = 0
        self.level_to_labels[0] = ['root']
        self.label_to_idx['root'] = 0
        self.label_to_abstract['root'] = ""
        self.label_to_title['root'] = ""
        self.label_has_description['root'] = 0
        self.idx_to_label[0] = 'root'
        # Only for the creation of the taxonomy, to keep track of all labels present
        self._all_labels = set()

        # Load and store the taxonomy
        # Root node is labeled as 'root'
        with open(self.cfg['paths']['dataset'] / 'ontology.json') as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                # Title of the label
                self.label_to_title[data['label']] = data['title']
                # Textual description
                if 'title' in data:
                    self.label_to_abstract[data['label']] = data['title']
                    self.label_has_description[data['label']] = 0
                if 'definition' in data:
                    if data['definition'] != "":
                        self.label_to_abstract[data['label']] = data['definition']
                        self.label_has_description[data['label']] = 1
                if 'txt' in data:
                    self.label_to_abstract[data['label']] = data['txt']
                    self.label_has_description[data['label']] = 1
                # Map label to its level in the taxonomy
                self.label_to_level[data['label']] = data['level']
                if data['level'] in self.level_to_labels:
                    self.level_to_labels[data['level']].append(data['label'])
                else:
                    self.level_to_labels[data['level']] = [data['label']]
                self._all_labels.add(data['label'])

        with open(self.cfg['paths']['dataset'] / 'taxonomy.txt') as f:
            for idx, line in enumerate(f):
                line = line.strip().replace('\n', '')
                parent, *children = line.split(' ')
                filtered_children = list(self._all_labels.intersection(children))
                # Name correctly the 'root' node
                if idx == 0:
                    parent = 'root'
                # Store the children of all nodes
                self.label_to_children[parent] = filtered_children

        # Be sure to have all leaves nodes in the taxonomy, i.e. also the ones having no children
        for label in self.label_to_title.keys():
            try:
                _ = self.label_to_children[label]
            except KeyError:
                self.label_to_children[label] = []

        # Check that labels taken from taxonomy and ontology match
        assert len(self.label_to_level) == len(self.label_to_children), \
            f"Lengths between taxonomy ({len(self.label_to_children)}) and ontology ({len(self.label_to_level)}) do not match"
        assert set(self.label_to_level.keys()) == set(self.label_to_children.keys()), \
            f"Labels do not match" 

        self.update_taxo(verbose=verbose)
        del self._all_labels


    # Update characteristics of the taxonomy, create a label to index and a label to parents mapping
    def update_taxo(self, verbose=False):
        # Store the parents of all nodes
        self.label_to_parents = {}
        self.label_to_parents['root'] = []
        for parent, children in self.label_to_children.items():
            for child in children:
                try:
                    # Add parent to child
                    self.label_to_parents[child].append(parent)
                except KeyError:
                    # Initialize with empty list
                    self.label_to_parents[child] = [parent]

        # Make sure there is no duplicate either in the children or in the parents
        for label in self.label_to_children.keys():
            self.label_to_children[label] = list(set(self.label_to_children[label]))
            self.label_to_parents[label] = list(set(self.label_to_parents[label]))

        # Update leaves and height
        self.leaves = sorted([label for label, children in self.label_to_children.items() if len(children) == 0])
        self.height = max(self.label_to_level.values())
        self.width = max( [len(level_labels) for level_labels in self.level_to_labels.values()] ) 
        self.n_nodes = len(self.label_to_children)
        self.n_edges = np.sum([len(children) for children in self.label_to_children.values()])
        
        # In a tree, all nodes have only one single parent
        # otherwise it is a semi-lattice
        self.is_a_tree = True
        for label, parents in self.label_to_parents.items():
            if label == 'root': continue
            if len(parents) > 1 and isinstance(parents, list):
                self.is_a_tree = False
                break

        # Used for classification, maps each label to an unique integer
        self.label_to_idx = {'root': 0}
        self.idx_to_label = {0: 'root'}
        all_children = [lab for lab in self.label_to_level.keys() if lab != 'root']
        for idx, label in enumerate(all_children):
            self.label_to_idx[label] = idx+1
            self.idx_to_label[idx+1] = label

        if verbose:
            self.print_stats()
            self.print_graph_stats()


    def print_stats(self):
        print(f"> Taxonomy information:")
        # Tree or semi-lattice
        if self.is_a_tree:
            print(f">> This taxonomy is a tree with {self.n_nodes} nodes and {np.sum([len(children) for children in self.label_to_children.values()])} edges")
        else:
            print(f">> This taxonomy is a semi-lattice graph with {self.n_nodes} nodes and {np.sum([len(children) for children in self.label_to_children.values()])} edges")

        # Number of leaves and levels, detail per level
        print(f">> It has {len(self.leaves)} leaves and {self.height} levels")
        for level, labels in self.level_to_labels.items():
            print(f">>> Level {level}: {len(labels)} labels ({len([lab for lab in labels if self.is_leaf(lab)])} are leaves)")

        # Number of descriptions available (-1 since we do not count the root)
        nodes_with_description = np.sum(list(self.label_has_description.values()))
        if self.label_has_description['root']: nodes_with_description -= 1
        print(f">> Nodes having a description (root excluded): {nodes_with_description}/{self.n_nodes-1} ({nodes_with_description/(self.n_nodes-1)*100:.2f}%)")


    def print_graph_stats(self):
        print(f"> Taxonomy graph information:")
        edge_list = self.get_edge_list()
        taxo_digraph = nx.DiGraph()
        taxo_digraph.add_edges_from(edge_list)
        try:
            # Need to keep orientation parent -> child for the edges
            cycle = nx.find_cycle(taxo_digraph, orientation='original')
        except nx.NetworkXNoCycle:
            pass
        else:
            assert False, f"Taxonomy graph contains a cycle: {cycle}"
        print(f">> Graph has {taxo_digraph.number_of_nodes()} nodes and {taxo_digraph.number_of_edges()} edges")
        print(f">> Graph is a tree -> {nx.is_tree(taxo_digraph)}")
        print(f">> Average degree -> {np.mean([deg for _, deg in taxo_digraph.degree()])}") # type: ignore
        print(f">> Density -> {nx.density(taxo_digraph)}")
        # Some properties can only be computed on undirected graphs
        taxo_graph = nx.Graph()
        taxo_graph.add_edges_from(edge_list)
        print(f">> Number of connected components -> {nxalgo.number_connected_components(taxo_graph)}")


    def is_leaf(self, label):
        return len(self.label_to_children[label]) == 0


    def all_children(self, label):
        self.children = set()
        self._recursive_all_children(label)
        # Sort so that every call is deterministic
        return sorted(list(self.children))


    def _recursive_all_children(self, label):
        if self.is_leaf(label):
            return
        for child in self.label_to_children[label]:
            self._recursive_all_children(child)
            self.children.add(child)


    def merge_nodes(self, label_kept, label_deleted):
        # Save label remapping
        self.label_remappings[label_deleted] = label_kept
        # Remove from label_to_level and level_to_labels
        self.level_to_labels[self.label_to_level[label_deleted]].remove(label_deleted)
        if len(self.level_to_labels[self.label_to_level[label_deleted]]) == 0:
            del self.level_to_labels[self.label_to_level[label_deleted]]
        del self.label_to_level[label_deleted]
        # Update states of children of removed node
        for child in self.label_to_children[label_deleted]:
            # Set new child to kept node
            self.label_to_children[label_kept].append(child)
            # Update level of the child
            self.level_to_labels[self.label_to_level[child]].remove(child)
            new_level = self.label_to_level[label_kept] + 1
            self.label_to_level[child] = new_level
            self.level_to_labels[new_level].append(child)

        # Delete all references to the deleted node
        for parent in self.label_to_parents[label_deleted]:
            self.label_to_children[parent].remove(label_deleted)
            # Parents of deleted node become parent of kept node
            if parent != label_kept:
                # Add the parent -> child relation according to the level in the taxonomy
                if self.label_to_level[parent] == self.label_to_level[label_kept] - 1:
                    self.label_to_children[parent].append(label_kept)
                elif self.label_to_level[parent] == self.label_to_level[label_kept] + 1:
                    self.label_to_children[label_kept].append(parent)
                else:
                    pass
        del self.label_to_parents[label_deleted]
        del self.label_to_children[label_deleted]
        del self.label_to_idx[label_deleted]
        del self.label_to_abstract[label_deleted]
        del self.label_to_title[label_deleted]
        del self.label_has_description[label_deleted]
        self.update_taxo(verbose=True)


    def remove_leaves(self, leaves):
        if not isinstance(leaves, list): leaves = [leaves]
        for leaf in leaves:
            assert self.is_leaf(leaf), f"Cannot remove {leaf}, not a leaf"
            # Remove from label_to_level and level_to_labels
            self.level_to_labels[self.label_to_level[leaf]].remove(leaf)
            if len(self.level_to_labels[self.label_to_level[leaf]]) == 0:
                del self.level_to_labels[self.label_to_level[leaf]]
            del self.label_to_level[leaf]
            # Delete label_to_children and update label_to_parents
            del self.label_to_children[leaf]
            for parent in self.label_to_parents[leaf]:
                self.label_to_children[parent].remove(leaf)
            del self.label_to_parents[leaf]
            del self.idx_to_label[self.label_to_idx[leaf]]
            del self.label_to_idx[leaf]
            del self.label_to_abstract[leaf]
            del self.label_to_title[leaf]
            del self.label_has_description[leaf]
            self.update_taxo()


    def get_subtaxonomy(self, new_root_label, verbose = True):
        # New object and get the new nodes (including the new root)
        subtaxonomy = Taxonomy()
        subtaxonomy.cfg = self.cfg
        new_nodes = ['root'] + self.all_children(new_root_label)
        root_level = self.label_to_level[new_root_label]

        # Compute all properties for the new taxonomy
        for idx, lab in enumerate(new_nodes):
            if lab == 'root':
                old_lab = new_root_label
                new_lab = 'root'
            else:
                old_lab = lab
                new_lab = lab

            # Properties with level
            subtaxonomy.label_to_level[new_lab] = self.label_to_level[old_lab]-root_level
            # Properties with indices
            subtaxonomy.label_to_idx[new_lab] = idx
            subtaxonomy.idx_to_label[idx] = new_lab
            # Properties with textual description
            subtaxonomy.label_to_abstract[new_lab] = self.label_to_abstract[old_lab]
            subtaxonomy.label_to_title[new_lab] = self.label_to_title[old_lab]
            subtaxonomy.label_has_description[new_lab] = self.label_has_description[old_lab]
            # Properties for the parent-children relations
            subtaxonomy.label_to_children[new_lab] = list(set([child for child in self.label_to_children[old_lab] if child in new_nodes]))
            if lab == 'root':
                subtaxonomy.label_to_parents[new_lab] = []
            else:
                subtaxonomy.label_to_parents[new_lab] = list(set([parent for parent in self.label_to_parents[old_lab] if parent in new_nodes]))

        # Compute label to level
        for label, level in subtaxonomy.label_to_level.items():
            try:
                subtaxonomy.level_to_labels[level].append(label)
            except KeyError:
                subtaxonomy.level_to_labels[level] = [label]

        subtaxonomy.update_taxo(verbose=verbose)

        return subtaxonomy

    def merge(self, other, verbose=True):
        """
        Merge two subtaxonomies (both with a synthetic 'root') into a new Taxonomy.

        Assumptions:
        - Both `self` and `other` are subtaxonomies obtained via `get_subtaxonomy(...)`
        from the same original taxonomy.
        - In both, the node called 'root' is the artificial root; its children are
        the actual roots of the respective subtrees we care about.

        The merged taxonomy:
        - Has a single 'root' node.
        - The children of 'root' are the union of the children of 'root' in `self` and `other`.
        - All other nodes/edges are the union of nodes/edges from both subtaxonomies.
        """

        assert isinstance(other, Taxonomy), "Can only merge with another Taxonomy"

        merged = Taxonomy()
        # Keep cfg from one of them (assuming same dataset)
        merged.cfg = getattr(self, "cfg", None)

        # Initialize root meta-information (like in load_taxonomy)
        merged.label_to_title['root'] = ""
        merged.label_to_abstract['root'] = ""
        merged.label_has_description['root'] = 0
        merged.label_to_level['root'] = 0
        merged.label_to_children['root'] = []

        # Helper to add all info from one subtaxonomy
        def _add_from(taxo: "Taxonomy"):
            # 1) Children / edges
            for parent, children in taxo.label_to_children.items():
                if parent == "root":
                    # its children become direct children of the global merged root
                    merged.label_to_children['root'].extend(children)
                    continue

                if parent not in merged.label_to_children:
                    merged.label_to_children[parent] = []
                merged.label_to_children[parent].extend(children)

            # 2) Levels
            for label, level in taxo.label_to_level.items():
                if label == "root":
                    continue
                if label in merged.label_to_level:
                    # sanity check: levels should be consistent
                    assert merged.label_to_level[label] == level, \
                        f"Level mismatch for label {label}: {merged.label_to_level[label]} vs {level}"
                else:
                    merged.label_to_level[label] = level

            # 3) Textual properties
            for label in taxo.label_to_title.keys():
                if label == "root":
                    continue
                if label in merged.label_to_title:
                    # If you want, you can assert equality here instead of silently skipping
                    # assert merged.label_to_title[label] == taxo.label_to_title[label]
                    continue
                merged.label_to_title[label] = taxo.label_to_title[label]
                merged.label_to_abstract[label] = taxo.label_to_abstract[label]
                merged.label_has_description[label] = taxo.label_has_description[label]

        # Add info from both subtaxonomies
        _add_from(self)
        _add_from(other)

        # Deduplicate children lists and make sure every label has a children list
        for label in list(merged.label_to_level.keys()):
            merged.label_to_children.setdefault(label, [])
            merged.label_to_children[label] = sorted(set(merged.label_to_children[label]))

        # Rebuild level_to_labels from merged.label_to_level
        merged.level_to_labels = {}
        for lab, lvl in merged.label_to_level.items():
            merged.level_to_labels.setdefault(lvl, []).append(lab)

        # Let the usual machinery recompute parents, leaves, idx maps, etc.
        merged.update_taxo(verbose=verbose)

        return merged

    # Extract the remapping when we merges nodes
    def extract_remapping(self, file_path):
        if not file_path.endswith(".json"): file_path += '.json'
        with open(file_path, 'w') as f:
            json.dump(self.label_remappings, f)


    # Extract the parent-child relation into a file so that it can be loaded later
    def extract_taxonomy(self, file_path):
        if not file_path.endswith(".txt"): file_path += '.txt'
        with open(file_path, 'w') as f:
            first_line = 'ROOT ' + ' '.join(self.label_to_children['root']) + '\n'
            f.write(first_line)
            for parent, children in self.label_to_children.items():
                if parent == 'root': continue
                if len(children) != 0:
                    str_line = str(parent) + ' ' + ' '.join(self.label_to_children[parent]) + '\n'
                    f.write(str_line)


    def get_edge_list(self):
        edge_list = []
        for parent, children in self.label_to_children.items():
            for child in children:
                edge_list.append((self.label_to_idx[parent], self.label_to_idx[child]))
        return edge_list
    
    def get_avg_degree(self, verbose = False):
        if verbose:
            print(f"> Taxonomy graph information:")
        edge_list = self.get_edge_list()
        taxo_digraph = nx.DiGraph()
        taxo_digraph.add_edges_from(edge_list)
        try:
            # Need to keep orientation parent -> child for the edges
            cycle = nx.find_cycle(taxo_digraph, orientation='original')
        except nx.NetworkXNoCycle:
            pass
        else:
            assert False, f"Taxonomy graph contains a cycle: {cycle}"
        return np.mean([deg for _, deg in taxo_digraph.degree()]) # type: ignore
    
    def get_degree_sum(self, verbose = False):
        if verbose:
            print(f"> Taxonomy graph information:")
        edge_list = self.get_edge_list()
        taxo_digraph = nx.DiGraph()
        taxo_digraph.add_edges_from(edge_list)
        try:
            # Need to keep orientation parent -> child for the edges
            cycle = nx.find_cycle(taxo_digraph, orientation='original')
        except nx.NetworkXNoCycle:
            pass
        else:
            assert False, f"Taxonomy graph contains a cycle: {cycle}"
        return np.sum([deg for _, deg in taxo_digraph.degree()]) # type: ignore
    
    def draw_taxonomy(self, file_name="taxonomy"):
        # Create the graph from edge list
        edge_list = self.get_edge_list()
        G = nx.DiGraph()
        G.add_edges_from(edge_list)
        # Hierarchical layout
        pos = self._semilattice_pos(G, root=self.label_to_idx['root'])
        plt.figure(figsize=(16,9))
        nx.draw(G, pos=pos, with_labels=True, node_size=900, node_color='red', linewidths=3)
        file_name_with_suffix = file_name + ".png"
        plt.savefig(self.cfg['paths']['output'] / file_name_with_suffix)
        plt.close()


    # Layout to draw the taxonomy with proper displayed levels
    # Inspired from https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209
    def _semilattice_pos(self, G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        """A layout function for plotting a semi-lattice (a directed acyclic graph where nodes may have multiple parents).

        Args:
            G (nx.DiGraph): The directed graph (must be a directed acyclic graph).
            root (str): The root node of the layout. If not provided, a root will be chosen.
            width (float): Horizontal space allocated for the layout.
            vert_gap (float): Gap between levels of the hierarchy.
            vert_loc (float): Vertical position of the root.
            xcenter(float): Horizontal position of the root.

        """
        if not nx.is_directed_acyclic_graph(G):
            raise TypeError('The function only supports directed acyclic graphs.')

        if root is None:
            # Attempt to find a root-like node (with no incoming edges)
            roots = [n for n in G.nodes if G.in_degree(n) == 0]
            if not roots:
                raise ValueError('The graph has no root-like nodes.')
            root = roots[0]

        def _calculate_positions(G, node, width=1., vert_gap=0.2, vert_loc=0., xcenter=0.5, pos=None, visited=None):
            """Helper function to recursively calculate positions for nodes.
            
            pos (dict): A dictionary storing positions of nodes.
            visited (set): A set to track visited nodes and avoid processing them multiple times.

            """
            if pos is None:
                pos = {}
            if visited is None:
                visited = set()

            if node in visited:
                return pos  # Avoid reprocessing nodes

            visited.add(node)
            pos[node] = (xcenter, vert_loc)

            children = list(G.successors(node))
            if children:
                dx = width / len(children)
                next_x = xcenter - width / 2 + dx / 2
                for child in children:
                    pos = _calculate_positions(G, child, width=dx, vert_gap=vert_gap, 
                                                vert_loc = vert_loc - vert_gap, xcenter=next_x, 
                                                pos=pos, visited=visited)
                    next_x += dx

            return pos

        # Start recursive position calculation
        pos = _calculate_positions(G, root, width=width, vert_gap=vert_gap, 
                                    vert_loc=vert_loc, xcenter=xcenter)

        return pos

    def is_relevant_to_taxonomy(self, labels):
        for label in labels :
            named_label = self.idx_to_label[int(label)]
            if named_label not in self.label_to_level :
                return False
        return True


    def labels_to_paths(self, labels):
        #named_labels = [self.idx_to_label[int(label)] for label in labels if int(label) in self.idx_to_label]
        named_labels = [label for label in labels if label in self.label_to_level]
        if len(named_labels) == 0 :
            return []
        
        

        pre_paths = [["root"]]
        label_per_level = {}

        output = []

        
        for label in named_labels :
            
            if label in self.label_to_level:
                level  = self.label_to_level[label]
                dlevel : list= label_per_level.get(level,[])
                dlevel.append(label)
                label_per_level[level] = dlevel

        max_level = max(label_per_level.keys())

        for level in range(1,max_level+1):
            if level not in label_per_level : continue
            
            post_paths = []
            for path in pre_paths :
                
                children = []
                for node in label_per_level[level] :
                    current_leaf = path[-1]
                    if current_leaf in self.label_to_children :
                        if node in self.label_to_children[current_leaf]:
                            post_paths.append(path+[node])
                            children.append(node)
                if len(children)>0 :
                    output.append( (path, children))
                    
            pre_paths = post_paths 

        #assert len(output)>0
        
        return output