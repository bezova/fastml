# testing loading the module
#from importlib import reload

#import fastml #__init__  should have : import fastml.utils
#fastml.utils.pp()

# import fastml #__init__  has: from .utils import pp
# fastml.utils.pp()

# import fastml #__init__  has: from fastml import utils
# fastml.utils.pp()

# from fastml import * # __init__ should have:  __all__ = ['utils']
# utils.pp()

# from fastml import * # __init__ should have:  from .utils import pp
# # next not important __all__ = ['pp']
# pp()


# from fastml import utils
# reload(utils)
# utils.pp()

# from fastml.utils import pp
# pp()

