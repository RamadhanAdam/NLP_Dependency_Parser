Assignment 1: Multilingual Dependency Parsing Name: Ramadhan Adam Zome

Part 1: Performance with badfeatures.model

LAS: 14.15%

UAS: 5.66%

Analysis: The performance is extremely poor because the provided 'badfeatures.model' uses a minimal feature set that lacks context. Transition-based parsing is a greedy process; without informative features about the stack and buffer state, the model cannot distinguish between SHIFT, LEFTARC, and RIGHTARC operations, leading to cascading errors.

Part 2a: Feature Extractor Implementation To improve performance, I expanded the feature set in featureextractor.py to include:

Context Window Expansion: Added Word, POS tag, and Lemma features for the top three items on the stack (s0, s1, s2) and the front of the buffer (b0, b1, b2). This allows the model to "look ahead" and "look back" at structural context.

Morphological Suffixes: Extracted the last 3 characters of words on s0 and b0. This is crucial for handling out-of-vocabulary (OOV) words and capturing grammatical markers in morphologically rich languages.

Structural Valency: Added features counting the current number of left and right dependents for the stack top (s0). This helps the model "know" when a head has already found its necessary dependents.

Distance Binning: Categorized the linear distance between s0 and b0 into bins (Adjacent, Near, Far). Longer distances often correlate with a lower probability of an immediate dependency arc.

Complexity: The feature extraction process is O(1) per transition. Because it relies on a fixed number of lookups from the stack and buffer, the extraction time does not increase with the length of the sentence, ensuring the overall parser remains linear.

Part 2c: Performance Results All models were trained using 200 sentences.

Language	LAS	UAS
English	0.7133	0.7474
Danish	0.7291	0.7991
Swedish	0.6504	0.7671
The English and Danish models exceeded the 70% LAS threshold. While the Swedish LAS is slightly below 70%, its UAS (76.71%) demonstrates high structural accuracy.

Part 2d: Arc-Eager Parser Complexity and Tradeoffs Complexity: The Arc-Eager shift-reduce parser operates in linear time O(n). Each token in a sentence of length n is pushed onto the stack exactly once and popped exactly once. This results in a maximum of 2n transitions.

Tradeoffs:

Speed vs. Global Optimality: The greedy nature makes it fast, but it lacks a global view of the tree, making it susceptible to error propagation.

Projectivity: The standard Arc-Eager algorithm can only produce projective trees (no crossing arcs).

Part 3: Parser Executable The parse.py script is implemented to support piping: cat <file> | python parse.py code/english.model > output.conll

It handles the necessary class loading for TransitionParser and FeatureExtractor and ensures the DependencyGraph validation is satisfied by providing a temporary root structure that is updated during the parse.