"""
Arc-Eager Transition-Based Dependency Parser
This module implements an arc-eager transition-based dependency parser using a linear classifier.
The parser learns to predict transition actions (SHIFT, REDUCE, LEFT-ARC, RIGHT-ARC) to build
dependency trees from input sentences.
The implementation follows the arc-eager parsing algorithm and uses an SGD classifier with
logistic loss for training. It supports feature extraction through user-defined functions
and handles projective dependency graphs.
Classes:
    Configuration: Represents the parser state including stack, buffer, and arcs
    TransitionParser: Main parser class that handles training and parsing
Key Features:
    - Arc-eager transition system
    - Linear classification with SGDClassifier
    - Feature extraction framework
    - Model persistence with joblib
    - Projectivity checking
    - Robust parsing with fallback mechanisms
Author: Dr. Mulang' Onando
    
"""
import copy 
import os 
import tempfile 
from typing import List, Tuple, Optional

import numpy as np 
import scipy.sparse as sparse 
from sklearn.datasets import load_svmlight_file 
from sklearn.linear_model import SGDClassifier

class Configuration(object): 
    """
    Class for holding configuration which is the partial analysis of the input sentence.
    The transition based parser aims at finding set of operators that transfer the initial
    configuration to the terminal configuration.

    The configuration includes:
        - Stack: for storing partially proceeded words
        - Buffer: for storing remaining input words
        - Set of arcs: for storing partially built dependency tree

    This class also provides a method to represent a configuration as list of features.
    """

    def __init__(self, dep_graph, feature_extractor):
        """
            :param dep_graph: the representation of an input in the form of dependency graph.
            :type dep_graph: DependencyGraph where the dependencies are not specified.
            :param feature_extractor: a function which operates on tokens, the
                stack, the buffer and returns a list of string features
        """

        # dep_graph.nodes contain list of token for a sentence
        self.stack = [0]    # The root element
        self.buffer = list(range(1, len(dep_graph.nodes))) # The rest is in the buffer
        # Built arcs: list of (head, relation, dependent)
        self.arcs = []
        self._tokens = dep_graph.nodes
        self._max_address = len(self.buffer)

        # user-supplied extractor: callable(tokens, buffer, stack, arcs) -> List[str]
        self._user_feature_extractor = feature_extractor

    def __str__(self):
        return "Stack: {} Buffer: {} Arcs: {}".format(self.stack, self.buffer, self.arcs)

    def extract_features(self):
        """
            Extracts features from the configuration
            :return: list(str)
        """
        return self._user_feature_extractor(
            self._tokens,
            self.buffer,
            self.stack,
            self.arcs
        )

class TransitionParser(object): 
    """ 
    An arc-eager transition-based parser trained with a linear classifier (SGDClassifier).
    It learns labels like: - SHIFT - REDUCE - LEFT-ARC:RELATION - RIGHT-ARC:RELATION
    and at parse time picks the best valid move according to the classifier scores.
    """

    def __init__(self, transition, feature_extractor, classifier="sgd"):
        self._dictionary = {}
        self._transition = {}
        self._match_transition = {}
        self._model = None
        self._user_feature_extractor = feature_extractor
        self.transitions = transition
        self._clf_type = classifier  # currently only 'sgd' implemented

    def _get_dep_relation(self, idx_parent, idx_child, depgraph):
        """_summary_

        Args:
            idx_parent (_type_): _description_
            idx_child (_type_): _description_
            depgraph (_type_): _description_

        Returns:
            _type_: _description_
        """
        p_node = depgraph.nodes[idx_parent]
        c_node = depgraph.nodes[idx_child]

        if c_node["word"] is None:
            return None  # Root

        if c_node.get("head") == p_node.get("address"):
            return c_node.get("rel")
        else:
            return None

    def _convert_to_binary_features(self, features: List[str]) -> str:
        """
        This function converts a feature into libsvm format, and adds it to the
        feature dictionary
        :param features: list of feature string which is needed to convert to
            binary features
        :type features: list(str)
        :return : string of binary features in libsvm format  which is
            'featureID:value' pairs
        """
        unsorted_result = []
        for feat in features:
            self._dictionary.setdefault(feat, len(self._dictionary))
            unsorted_result.append(self._dictionary[feat])

        return " ".join(
            str(fid) + ":1.0" for fid in sorted(unsorted_result)
        )

    @staticmethod
    def _is_projective(depgraph) -> bool:
        """
        Projectivity check for dependency graphs.
        """
        arc_list = set()
        for key in depgraph.nodes:
            node = depgraph.nodes[key]
            if "head" in node:
                childIdx = node["address"]
                parentIdx = node["head"]
                arc_list.add((parentIdx, childIdx))

        for parentIdx, childIdx in arc_list:
            # Ensure that childIdx < parentIdx
            if childIdx > parentIdx:
                childIdx, parentIdx = parentIdx, childIdx
            for k in range(childIdx + 1, parentIdx):
                for m in range(len(depgraph.nodes)):
                    if (m < childIdx) or (m > parentIdx):
                        if (k, m) in arc_list:
                            return False
                        if (m, k) in arc_list:
                            return False
        return True

    def _write_to_file(self, key: str, binary_features: str, input_file):
        """
        write the binary features to input file and update the transition dictionary
        :param key: transition key
        :type key: str
        :param binary_features: features in libsvm format
        :type binary_features: str
        :param input_file: file object to write the features
        :type input_file: file object"""
        self._transition.setdefault(key, len(self._transition) + 1)
        self._match_transition[self._transition[key]] = key
        input_str = str(self._transition[key]) + " " + binary_features + "\n"
        input_file.write(input_str.encode("utf-8"))

    def _create_training_examples_arc_eager(self, depgraphs, input_file):
        """
        Create the training example in the libsvm format and write it to the input_file.
        Reference : 'A Dynamic Oracle for Arc-Eager Dependency Parsing' by Joav Goldberg and Joakim Nivre
        :param depgraphs: list of dependency graphs for training
        :type depgraphs: list(DependencyGraph)
        :param input_file: file object to write the features
        :type input_file: file object
        :return: list of transition keys in the order they are generated
        :rtype: list(str)
        """
        training_seq = []
        projective_graphs = [dg for dg in depgraphs if TransitionParser._is_projective(dg)]
        count_proj = len(projective_graphs)

        for depgraph in projective_graphs:
            conf = Configuration(depgraph, self._user_feature_extractor.extract_features)

            while conf.buffer:
                b0 = conf.buffer[0]
                features = conf.extract_features()
                binary_features = self._convert_to_binary_features(features)

                if conf.stack:
                    s0 = conf.stack[-1]
                    # Left-arc
                    rel = self._get_dep_relation(b0, s0, depgraph)
                    if rel is not None:
                        key = self.transitions.LEFT_ARC + ":" + rel
                        self._write_to_file(key, binary_features, input_file)
                        self.transitions.left_arc(conf, rel)
                        training_seq.append(key)
                        continue

                    # Right-arc
                    rel = self._get_dep_relation(s0, b0, depgraph)
                    if rel is not None:
                        key = self.transitions.RIGHT_ARC + ":" + rel
                        self._write_to_file(key, binary_features, input_file)
                        self.transitions.right_arc(conf, rel)
                        training_seq.append(key)
                        continue

                    # Reduce
                    flag = False
                    for k in range(s0):
                        if self._get_dep_relation(k, b0, depgraph) is not None:
                            flag = True
                        if self._get_dep_relation(b0, k, depgraph) is not None:
                            flag = True

                    if flag:
                        key = self.transitions.REDUCE
                        self._write_to_file(key, binary_features, input_file)
                        self.transitions.reduce(conf)
                        training_seq.append(key)
                        continue

                # Default: shift
                key = self.transitions.SHIFT
                self._write_to_file(key, binary_features, input_file)
                self.transitions.shift(conf)
                training_seq.append(key)

        print("Number of sentences: {}".format(len(depgraphs)))
        print("Number of valid (projective) sentences: {}".format(count_proj))
        print("Number of transition instances: {}".format(len(training_seq)))
        return training_seq

    def train(self, depgraphs):
        """
        Train an SGDClassifier (logistic loss) on generated transition instances.
        """
        try:
            with tempfile.NamedTemporaryFile(prefix="train", delete=False) as tf:
                self._create_training_examples_arc_eager(depgraphs, tf)
                temp_name = tf.name

            x_train, y_train = load_svmlight_file(temp_name, dtype=np.float64)
            # Make sure indices/indptr are expected types
            if sparse.isspmatrix_csr(x_train) or sparse.isspmatrix_csc(x_train):
                x_train.indices = x_train.indices.astype(np.int32, copy=False)
                x_train.indptr = x_train.indptr.astype(np.int32, copy=False)
            y_train = y_train.astype(np.int64, copy=False)

            # Use SGD with log-loss for OvR probabilities
            self._model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=2000,
                tol=1e-4,
                random_state=42
            )

            print("Training classifier (SGD, log_loss)...")
            self._model.fit(x_train, y_train)
            print("Training complete.")
        finally:
            if 'temp_name' in locals() and os.path.exists(temp_name):
                os.remove(temp_name)

    # ---------------- Parsing helpers ----------------

    @staticmethod
    def _has_head(idx: int, arcs: List[Tuple[int, str, int]]) -> bool:
        for (h, r, d) in arcs:
            if d == idx:
                return True
        return False

    def _can_left_arc(self, conf: Configuration) -> bool:
        if not conf.stack or not conf.buffer:
            return False
        s0 = conf.stack[-1]
        # Cannot assign a head to ROOT (0)
        if s0 == 0:
            return False
        # Left-arc requires the stack top to not already have a head
        if self._has_head(s0, conf.arcs):
            return False
        return True

    def _can_right_arc(self, conf: Configuration) -> bool:
        return bool(conf.stack and conf.buffer)

    def _can_reduce(self, conf: Configuration) -> bool:
        if not conf.stack:
            return False
        s0 = conf.stack[-1]
        return self._has_head(s0, conf.arcs)

    def _can_shift(self, conf: Configuration) -> bool:
        return bool(conf.buffer)

    @staticmethod
    def _decode_key(key: str) -> Tuple[str, Optional[str]]:
        # "LEFT-ARC:rel" -> ("LEFT-ARC", "rel"), "SHIFT" -> ("SHIFT", None)
        if ":" in key:
            a, rel = key.split(":", 1)
            return a, rel
        return key, None

    def _apply_action(self, conf: Configuration, action: str, rel: Optional[str]):
        if action == self.transitions.SHIFT:
            self.transitions.shift(conf)
        elif action == self.transitions.REDUCE:
            self.transitions.reduce(conf)
        elif action == self.transitions.LEFT_ARC:
            # rel is required
            self.transitions.left_arc(conf, rel or "dep")
        elif action == self.transitions.RIGHT_ARC:
            self.transitions.right_arc(conf, rel or "dep")
        else:
            # Unknown action: fallback to shift if possible
            if self._can_shift(conf):
                self.transitions.shift(conf)

    def _is_action_valid(self, conf: Configuration, action: str) -> bool:
        if action == self.transitions.SHIFT:
            return self._can_shift(conf)
        if action == self.transitions.REDUCE:
            return self._can_reduce(conf)
        if action == self.transitions.LEFT_ARC:
            return self._can_left_arc(conf)
        if action == self.transitions.RIGHT_ARC:
            return self._can_right_arc(conf)
        return False

    def _feature_row_sparse(self, features: List[str]):
        # Convert features to 1xV CSR sparse vector
        cols = []
        for feat in features:
            if feat in self._dictionary:
                cols.append(self._dictionary[feat])
        if not cols:
            # No known features; return empty but with correct shape
            return sparse.csr_matrix((1, len(self._dictionary)), dtype=np.float64)

        cols = np.array(sorted(cols), dtype=np.int32)
        rows = np.zeros_like(cols, dtype=np.int32)
        data = np.ones_like(cols, dtype=np.float64)
        x = sparse.csr_matrix((data, (rows, cols)), shape=(1, len(self._dictionary)), dtype=np.float64)
        x.indices = x.indices.astype(np.int32, copy=False)
        x.indptr = x.indptr.astype(np.int32, copy=False)
        return x

    def _finalize_graph_with_arcs(self, depgraph, arcs: List[Tuple[int, str, int]]):
        # Clear heads/rels except root
        for i in range(1, len(depgraph.nodes)):
            depgraph.nodes[i]["head"] = None
            depgraph.nodes[i]["rel"] = None
        # Apply predicted arcs
        for (h, r, d) in arcs:
            if d < len(depgraph.nodes):
                depgraph.nodes[d]["head"] = h
                depgraph.nodes[d]["rel"] = r if r is not None else "dep"
        # Any token still without head? attach to ROOT
        for i in range(1, len(depgraph.nodes)):
            if depgraph.nodes[i].get("head") is None:
                depgraph.nodes[i]["head"] = 0
                depgraph.nodes[i]["rel"] = depgraph.nodes[i].get("rel") or "dep"

    def parse(self, depgraphs):
        """
        Parse a list of dependency graphs using the trained classifier.
        Returns a list of predicted dependency graphs (with heads and rels filled).
        """
        if not self._model:
            raise ValueError("No model trained!")

        results = []
        m = self._model
        classes = m.classes_
        # Mapping class index -> label id -> transition key
        class_index_to_key = {i: self._match_transition[c] for i, c in enumerate(classes)}

        for gold_graph in depgraphs:
            # Work on a fresh copy, with heads/rel cleared
            depgraph = copy.deepcopy(gold_graph)
            for i in range(1, len(depgraph.nodes)):
                depgraph.nodes[i]["head"] = None
                depgraph.nodes[i]["rel"] = None

            conf = Configuration(depgraph, self._user_feature_extractor.extract_features)

            # Continue until buffer empty and only ROOT remains on stack
            safety = 5 * len(depgraph.nodes) + 10  # prevent infinite loops
            while (conf.buffer or len(conf.stack) > 1) and safety > 0:
                safety -= 1
                
                feats = conf.extract_features()
                x_test = self._feature_row_sparse(feats)

                # Get scores
                if hasattr(m, "predict_proba"):
                    scores = m.predict_proba(x_test)[0]  # aligned with classes_
                else:
                    s = m.decision_function(x_test)
                    scores = np.ravel(s)

                # Rank classes from best to worst
                ranked = np.argsort(-scores)

                applied = False
                for idx in ranked:
                    key = class_index_to_key.get(idx)
                    if key is None:
                        continue
                    action, rel = self._decode_key(key)
                    if self._is_action_valid(conf, action):
                        self._apply_action(conf, action, rel)
                        applied = True
                        break

                # Fallbacks to guarantee progress
                if not applied:
                    if self._can_shift(conf):
                        self.transitions.shift(conf)
                    elif self._can_reduce(conf):
                        self.transitions.reduce(conf)
                    else:
                        # If nothing can be done, break to avoid infinite loop
                        break

            # Build the final graph from conf.arcs
            self._finalize_graph_with_arcs(depgraph, conf.arcs)
            results.append(depgraph)

        return results

    def save(self, filepath):
        """
        Save model and mappings with joblib.
        """
        from joblib import dump
        bundle = {
            "model": self._model,
            "dictionary": self._dictionary,
            "transition": self._transition,
            "match_transition": self._match_transition,
        }
        dump(bundle, filepath, compress=3)
        print("Model bundle saved to {}".format(filepath))

    @staticmethod
    def load(filepath, transition_class, feature_extractor_class):
        """
        Load model and mappings (works across Python 3.x without libsvm quirks).
        """
        from joblib import load as joblib_load
        bundle = joblib_load(filepath)

        tp = TransitionParser(transition_class, feature_extractor_class, classifier="sgd")
        tp._model = bundle["model"]
        tp._dictionary = bundle["dictionary"]
        tp._transition = bundle["transition"]
        tp._match_transition = bundle["match_transition"]
        return tp
if __name__ == "__main__":
    import argparse
    from dependencygraph import DependencyGraph
    from transition import Transition
    from featureextractor import FeatureExtractor

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-p", "--parse", action="store_true")
    parser.add_argument("-i", "--input", help="Input CoNLL file")
    parser.add_argument("-m", "--model", help="Model file path")
    parser.add_argument("-o", "--output", help="Output file path (for parsing)")
    args = parser.parse_args()

    sub_extractor = FeatureExtractor()

    if args.train:
        with open(args.input, 'r', encoding='utf-8') as f:
            train_graphs = [DependencyGraph(sent) for sent in f.read().split('\n\n') if sent.strip()]
        
        tp = TransitionParser(Transition, sub_extractor)
        tp.train(train_graphs)
        tp.save(args.model)

    elif args.parse:
        tp = TransitionParser.load(args.model, Transition, sub_extractor)
        
        with open(args.input, 'r', encoding='utf-8') as f:
            test_graphs = [DependencyGraph(sent) for sent in f.read().split('\n\n') if sent.strip()]

        results = tp.parse(test_graphs)

        with open(args.output, 'w', encoding='utf-8') as f:
            for graph in results:
                f.write(graph.to_conll(10) + '\n\n')
        
        print(f"Successfully saved parsed results to {args.output}")