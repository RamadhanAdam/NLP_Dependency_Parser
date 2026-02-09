# Multilingual Dependency Parsing
**Author:** Ramadhan Adam Zome  
**Course:** Natural Language Processing

## Project Overview
This project implements a transition-based dependency parser using Nivre’s Arc-Eager shift-reduce algorithm. The parser is designed to predict grammatical relationships for English, Danish, and Swedish datasets in the CoNLL-U format.

---

## Directory Structure
The project is organized as follows:

```text
Assignment1/
├── code/
│   ├── english.model          # Trained English model
│   ├── danish.model           # Trained Danish model
│   ├── swedish.model          # Trained Swedish model
│   ├── featureextractor.py    # Custom feature engineering
│   ├── transition.py          # Transition operations logic
│   └── providedcode/          # Library for parsing and evaluation
├── data/                      # Training and test datasets
├── parse.py                   # Parser executable (supports piping)
├── README.md                  # Project documentation (Markdown)
├── README.txt                 # Project documentation (Plain Text)
└── englishfile                # Sample input for testing
```

---

## Performance Analysis

### Part 1: Baseline (badfeatures.model)
* **LAS:** 14.15%
* **UAS:** 5.66%

**Analysis:** Performance is extremely poor because the baseline uses a minimal feature set that lacks context. Transition-based parsing is a greedy process; without informative features about the stack and buffer state, the model cannot distinguish between SHIFT, LEFTARC, and RIGHTARC operations, leading to cascading errors.

### Part 2a: Feature Extractor Implementation
The `featureextractor.py` includes the following features:

1. **Context Window Expansion:** Word, POS tag, and Lemma features for the top three items on the stack ($s_0, s_1, s_2$) and the front of the buffer ($b_0, b_1, b_2$).
2. **Morphological Suffixes:** Last 3 characters of words on $s_0$ and $b_0$ to handle out-of-vocabulary (OOV) words.
3. **Structural Valency:** Current number of left and right dependents for the stack top ($s_0$).
4. **Distance Binning:** Linear distance between $s_0$ and $b_0$ categorized into bins (Adjacent, Near, Far).

**Complexity:** The feature extraction process is $O(1)$ per transition. It relies on a fixed number of lookups, ensuring extraction time remains constant regardless of sentence length.

### Part 2c: Performance Results (200 Training Sentences)
| Language | LAS | UAS |
| :--- | :--- | :--- |
| English | 0.7133 | 0.7474 |
| Danish | 0.7291 | 0.7991 |
| Swedish | 0.6504 | 0.7671 |

The English and Danish models exceeded the 70% LAS threshold. The Swedish model achieved a UAS of 76.71%, demonstrating high structural accuracy.

---

## Algorithm: Arc-Eager Shift-Reduce

### Complexity
The Arc-Eager shift-reduce parser operates in linear time $O(n)$. Each token in a sentence of length $n$ is pushed onto the stack once and popped once, resulting in a maximum of $2n$ transitions.

### Tradeoffs
* **Speed vs. Global Optimality:** The greedy nature is fast and efficient but lacks a global view of the tree, making it susceptible to error propagation.
* **Projectivity:** The algorithm is limited to producing projective dependency trees (no crossing arcs).

---

## Usage
The `parse.py` script supports standard input piping:

```bash
cat englishfile | python parse.py english.model > englishfile.conll
```