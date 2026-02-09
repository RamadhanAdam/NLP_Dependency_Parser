import sys
import os

# Add 'code' directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code')))

from providedcode.transitionparser import TransitionParser
from providedcode.dependencygraph import DependencyGraph
from featureextractor import FeatureExtractor
from transition import Transition # Crucial import for action constants

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    model_path = sys.argv[1]

    # 1. Load the model
    tp = TransitionParser.load(model_path, TransitionParser, FeatureExtractor)
    
    # 2. FIX: Re-link the transition constants to the loaded instance
    # This prevents the AttributeError: 'SHIFT'
    tp.transitions = Transition

    for line in sys.stdin:
        sentence = line.strip()
        if not sentence:
            continue
        
        words = sentence.split()
        
        # 3. Create CoNLL string that satisfies the root element check
        # The library's DependencyGraph is very picky about Column 7 (HEAD) and 8 (REL)
        conll_lines = []
        for i, word in enumerate(words):
            idx = i + 1
            # First word (1) depends on ROOT (0) with label 'ROOT'
            # Others depend on the first word temporarily
            head = "0" if idx == 1 else "1"
            rel = "ROOT" if idx == 1 else "dep"
            conll_lines.append(f"{idx}\t{word}\t_\t_\t_\t_\t{head}\t{rel}\t_\t_")
        
        conll_text = "\n".join(conll_lines) + "\n"

        try:
            # 4. Initialize graph and parse
            dg = DependencyGraph(conll_text)
            parsed = tp.parse([dg])
            
            # 5. Output the result in 10-column format
            if parsed:
                print(parsed[0].to_conll(10))
                print() # Extra newline for CoNLL readability
        except Exception as e:
            # If the graph validation still fails, we'll see exactly why here
            sys.stderr.write(f"Error: {str(e)}\n")