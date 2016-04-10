"""
Custom bbox interface class
"""

def get_bbox(name=None):
    """
    Get bbox instance module by name. Name == None loads original bbox module. All others must
    match a module from custom package.
    :param name:
    :return:
    """
    if name is None:
        print "We'll use original bbox implementation"
        import interface as bbox
        return bbox

    print "Will use bbox from module custom." + name
    res = __import__("lib.custom." + name)
    return getattr(getattr(res, "custom"), name)

