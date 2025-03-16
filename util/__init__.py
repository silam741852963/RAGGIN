import math

def normalize_distance(distance):
    """
    Normalize a given raw distance using an arctan transformation.
    
    The formula used is:
        normalized = (pi/2 - atan(distance)) / (pi/2)
        
    This maps:
      - A raw distance of 0 to 1 (highest relevance).
      - A raw distance tending to +infinity to 0.
      - A raw distance tending to -infinity to a value greater than 1, which is then clamped to 1.
    
    Returns a value in the range [0, 1].
    """
    val = (math.pi / 2 - math.atan(distance)) / (math.pi / 2)
    return max(0, min(val, 1))