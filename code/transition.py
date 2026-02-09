# -*- coding: utf-8 -*-


class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """

    # Define set of transitions
    LEFT_ARC = "LEFTARC"
    RIGHT_ARC = "RIGHTARC"
    SHIFT = "SHIFT"
    REDUCE = "REDUCE"

    def __init__(self):
        raise ValueError("Do not construct this object!")

    @staticmethod
    def left_arc(conf, relation):
        """
        LEFT-ARC: Add arc from buffer[0] (head) to stack top (dependent), then pop stack.
        Preconditions: stack is not empty, buffer is not empty, stack top is not ROOT
        """
        # Need buffer and stack to both have elements
        if len(conf.stack) < 1 or len(conf.buffer) < 1:
            return -1
        
        # Stack top cannot be ROOT (index 0)
        if conf.stack[-1] == 0:
            return -1
        
        # Create arc: buffer[0] (head) -> stack top (dependent)
        head = conf.buffer[0]
        dependent = conf.stack.pop()
        
        conf.arcs.append((head, relation, dependent))
        
        return conf

    @staticmethod
    def right_arc(conf, relation):
        """Add the arc (s, L, b) to A (arcs), and push b onto Σ.

        :param conf: is the current configuration
        :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        # You get this one for free! Use it as an example.

        s = conf.stack[-1]
        # pop the buffer
        b = conf.buffer.pop(0)

        conf.stack.append(b)
        conf.arcs.append((s, relation, b))

        # forgot to return conf
        return conf

    @staticmethod
    def shift(conf):
        """
        :param conf: is the current configuration
        :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        # Check : if buffer is empty
        if len(conf.buffer) == 0:
            return -1
        
        # Move buffer to stack
        b = conf.buffer.pop(0)
        conf.stack.append(b)
        return conf

    @staticmethod
    def reduce(conf):
        """
        :param conf: is the current configuration
        :return : A new configuration or -1 if the pre-condition is not satisfied
        """

        # Check : stack is empty
        if len(conf.stack) == 0:
            return -1

        # Check : stack top must have a head before popping( appear dependent in some arc)
        s = conf.stack[-1]
        has_head = any(arc[2] == s for arc in conf.arcs)

        if not has_head:
            return -1

        # Pop stack
        conf.stack.pop()
        return conf

