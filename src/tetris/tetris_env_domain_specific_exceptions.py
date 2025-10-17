class EmptyContainerError(Exception):
    """
    A custom, domain-specific exception for when an element
    of a container-object is about to be accessed /
    trying to be accessed, and the container-object is empty.
    """

    pass

class NoneTypeError(Exception):
    """
    A custom, domain-specific exception for when an object
    is of type "None", although it was expected not to be
    of type "None".
    """

    pass


class WrongDatatypeError(Exception):
    """
    A custom, domain-specific exception for when the datatype
    of an object passed to an argument of a function doesn't fit the
    expected datatype of this argument.
    """

    pass


class OutOfBoundsError(Exception):
    """
    A custom, domain-specific exception for when a tile reaches
    out of one or more borders of the field.
    """

    pass
