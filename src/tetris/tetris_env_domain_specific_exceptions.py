

class EmptyContainerError(Exception):
    """
    A custom, domain-specific exception for when an element
    of a container-object is about to be accessed / 
    trying to be accessed, and the container-object is empty.
    """

    pass