

class Namespace:
    '''
    Generic namespace class that can be set by a dictionary object
    and return None for missing arguments
    '''
    def __init__(self, params):
        for name in params:
            setattr(self, name, params[name])

    def __eq__(self, other):
        if not isinstance(other, Namespace):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.__dict__

    def __getattr__(self, name):
        return None
