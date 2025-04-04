def calc_span_f1_strict(span_1, span_2) -> int:
    """
    Calculate the strict F1 score for two spans
    if the spans have same start and end and label -> 1 
    otherwise -> 0 
    """
    s1, e1, l1 = span_1
    s2, e2, l2 = span_2

    #return 1 if spans line up and label is same 
    if s1 == s2 and e1 == e2 and l1 == l2:
        return 1
    
    #return zero otherwise
    return 0


def calc_span_f1_relaxed(span_1, span_2)->float:
    """
    Calculate the overlap between two spans with the same label
    if the labels are different -> 0
    if the spans do not overlap -> 0
    if the spans overlap -> overlap / union_length (value between 0 and 1)
    """
    s1, e1, l1 = span_1
    s2, e2, l2 = span_2

    #return 0 if labels are not the same 
    if l1 != l2:
        return 0
    
    # get the overlap
    overlap_start = max(s1, s2)
    overlap_end = min(e1, e2)

    if overlap_start <= overlap_end:
        union_length = max(e1, e2) - min(s1, s2)
        return (overlap_end - overlap_start) / union_length
    
    #return 0 if no overlap 
    return 0
    