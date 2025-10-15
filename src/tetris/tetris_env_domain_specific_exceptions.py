class EmptyContainerError(Exception):
    """
    A custom, domain-specific exception for when an element
    of a container-object is about to be accessed /
    trying to be accessed, and the container-object is empty.
    """

    pass

class WrongDatatypeError(Exception):
    """
    A custom, domain-specific exception for when the datatype
    of an object passed to an argument of a function doesn't fit the
    expected datatype of this argument.
    """
    
    pass