class FeatureExtractor(object): 
    @staticmethod
    def _check_informative(feat, underscore_is_informative=False):
        if feat is None:
            return False
        if feat == "":
            return False
        if not underscore_is_informative and feat == "_":
            return False
        return True

    @staticmethod
    def find_left_right_dependencies(idx, arcs):
        left_most = 10**6
        right_most = -1
        dep_left_most = ""
        dep_right_most = ""
        for wi, r, wj in arcs:
            if wi == idx:
                if (wj > wi) and (wj > right_most):
                    right_most = wj
                    dep_right_most = r
                if (wj < wi) and (wj < left_most):
                    left_most = wj
                    dep_left_most = r
        return dep_left_most, dep_right_most

    @staticmethod
    def extract_features(tokens, buffer, stack, arcs):
        result = []

        # 1. STACK POS/WORD/LEMMA (s0, s1, s2)
        for i, name in enumerate(['s0', 's1', 's2']):
            if len(stack) > i:
                idx = stack[-(i + 1)]
                token = tokens[idx]
                if FeatureExtractor._check_informative(token.get("ctag")):
                    result.append(f"{name}_pos={token['ctag']}")
                if i == 0:
                    if FeatureExtractor._check_informative(token.get("word")):
                        result.append(f"s0_word={token['word']}")
                    if FeatureExtractor._check_informative(token.get("lemma")):
                        result.append(f"s0_lemma={token['lemma']}")
            else:
                result.append(f"{name}_pos=NULL")

        # 2. BUFFER POS/WORD (b0, b1, b2)
        for i, name in enumerate(['b0', 'b1', 'b2']):
            if len(buffer) > i:
                idx = buffer[i]
                token = tokens[idx]
                if FeatureExtractor._check_informative(token.get("ctag")):
                    result.append(f"{name}_pos={token['ctag']}")
                if i < 2:
                    if FeatureExtractor._check_informative(token.get("word")):
                        result.append(f"{name}_word={token['word']}")
            else:
                result.append(f"{name}_pos=NULL")

        # 3. BIGRAM POS (s0 + b0)
        if stack and buffer:
            s0_p = tokens[stack[-1]].get("ctag")
            b0_p = tokens[buffer[0]].get("ctag")
            if FeatureExtractor._check_informative(s0_p) and FeatureExtractor._check_informative(b0_p):
                result.append(f"s0b0_pos={s0_p}+{b0_p}")

        # 4. TRIGRAM POS (s1 + s0 + b0)
        if len(stack) > 1 and buffer:
            s1_p = tokens[stack[-2]].get("ctag")
            s0_p = tokens[stack[-1]].get("ctag")
            b0_p = tokens[buffer[0]].get("ctag")
            if all(FeatureExtractor._check_informative(p) for p in [s1_p, s0_p, b0_p]):
                result.append(f"s1s0b0_pos={s1_p}+{s0_p}+{b0_p}")

        # 5. CHILD RELATIONS
        if stack:
            s0_idx = stack[-1]
            lc_rel, rc_rel = FeatureExtractor.find_left_right_dependencies(s0_idx, arcs)
            if FeatureExtractor._check_informative(lc_rel):
                result.append(f"s0_left_rel={lc_rel}")
            if FeatureExtractor._check_informative(rc_rel):
                result.append(f"s0_right_rel={rc_rel}")

        # 6. DISTANCE
        if stack and buffer:
            dist = abs(buffer[0] - stack[-1])
            dist_bin = "adj" if dist == 1 else "near" if dist <= 3 else "far"
            result.append(f"dist_bin={dist_bin}")
        # 7. MORPHOLOGY (Suffixes for Danish/Swedish)
        # Suffixes help with out-of-vocabulary words.
        if stack:
            s0_word = tokens[stack[-1]].get("word")
            if FeatureExtractor._check_informative(s0_word):
                result.append(f"s0_suffix3={s0_word[-3:]}")
        
        if buffer:
            b0_word = tokens[buffer[0]].get("word")
            if FeatureExtractor._check_informative(b0_word):
                result.append(f"b0_suffix3={b0_word[-3:]}")

        # 8. VALENCY (Child Counting)
        # Counts how many dependents s0 already has.
        if stack:
            s0_idx = stack[-1]
            # Simple count of arcs where s0 is the head
            num_deps = len([1 for wi, r, wj in arcs if wi == s0_idx])
            result.append(f"s0_valency={num_deps}")

        # 9. BUFFER POS BIGRAM (b0 + b1)
        # Helps anticipate if the buffer starts with a phrase (e.g., Adj + Noun)
        if len(buffer) > 1:
            b0_p = tokens[buffer[0]].get("ctag")
            b1_p = tokens[buffer[1]].get("ctag")
            if FeatureExtractor._check_informative(b0_p) and FeatureExtractor._check_informative(b1_p):
                result.append(f"b0b1_pos={b0_p}+{b1_p}")
                
        return result