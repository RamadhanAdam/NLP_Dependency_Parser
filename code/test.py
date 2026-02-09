"""
Test script for dependency parsing using transition-based parsing algorithm.

This script demonstrates the training and evaluation pipeline for a dependency parser
that uses arc-eager transition-based parsing with feature extraction and machine learning.

The script performs the following steps:
1. Loads training data from English corpus
2. Trains a transition parser with SGD classifier (commented out by default)
3. Loads a pre-trained model
4. Parses test data using the loaded model
5. Outputs results in CoNLL format
6. Evaluates parsing performance using LAS and UAS metrics

Author: Dr. Mulang' Onando
Date: January 2026
Course: Introduction to Natural Language Processing
Assignment: Assignment 1 - Dependency Parsing

Dependencies:
    - featureextractor: Contains FeatureExtractor class for feature engineering
    - providedcode.dataset: Provides access to training/test corpora
    - providedcode.evaluate: Contains DependencyEvaluator for performance metrics
    - providedcode.transitionparser: Core transition parser implementation
    - transition: Contains Transition class for parsing operations

Expected Output:
    - Parsed sentences in CoNLL format (test.conll)
    - LAS (Labeled Attachment Score) and UAS (Unlabeled Attachment Score) metrics
    
Note:
    The script includes error handling for NotImplementedError exceptions that occur
    when required methods in transition.py or featureextractor.py are not implemented.
"""

import random
from featureextractor import FeatureExtractor
from providedcode import dataset
from providedcode.evaluate import DependencyEvaluator
from providedcode.transitionparser import TransitionParser
from transition import Transition

if __name__ == "__main__": 
    data = dataset.get_english_train_corpus().parsed_sents()
    random.seed(1234)
    subdata = random.sample(data, 200)

    try:
        # Train [For testing purposes, there is already a badfeaturesmodel saved, so you can skip this step if you want]
        # For the Assignment, you need to uncomment this part to train
        tp = TransitionParser(Transition, FeatureExtractor, classifier="sgd")
        # tp.train(subdata)
        # tp.save("code/badfeaturesmodel.joblib")

        print("Model saved to code/badfeaturesmodel.joblib")

        # Load
        testdata = dataset.get_english_test_corpus().parsed_sents()
        tp = TransitionParser.load("code/badfeaturesmodel.joblib", Transition, FeatureExtractor)

        # Parse
        parsed = tp.parse(testdata)

        # Write CoNLL output as text (not bytes)
        with open("test.conll", "w", encoding="utf-8") as f:
            for p in parsed:
                f.write(p.to_conll(10))
                f.write("\n")

        # Evaluate
        ev = DependencyEvaluator(testdata, parsed)
        las, uas = ev.eval()
        print("LAS: {} \nUAS: {}".format(las, uas))

    except NotImplementedError as e:
        print("""
        This file is currently broken! We removed the implementation of Transition
        (in transition.py), which tells the transitionparser how to go from one
        Configuration to another Configuration. This is an essential part of the
        arc-eager dependency parsing algorithm, so you should probably fix that :)

        The algorithm is described in great detail here:
            http://aclweb.org/anthology//C/C12/C12-1059.pdf

        We also haven't actually implemented most of the features for for the
        support vector machine (in featureextractor.py), so as you might expect the
        evaluator is going to give you somewhat bad results...

        Your output should look something like this:

            LAS: 0.23023302131
            UAS: 0.125273849831

        Not this:

            Traceback (most recent call last):
                File "test.py", line 41, in <module>
                    ...
                    NotImplementedError: Please implement shift!
        {}
        """.format(e))
