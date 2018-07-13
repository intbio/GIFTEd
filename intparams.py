class newDict():
    def __init__(self):
        self.dict = {}
    def __setitem__(self, key, item):
        self.dict[tuple(key)] = item
    def __getitem__(self, key):
        key = tuple(key)
        revkey = tuple(reversed(key))
        if key in self.dict:
            return self.dict[key]
        elif revkey in self.dict:
            return self.dict[revkey]
    def __contains__(self, item):
        if item in self.dict or tuple(reversed(item)) in self.dict:
            return True
        else:
            return False
    def __iter__(self):
        for x in self.dict:
            yield x

class atomType:
    def __init__(self, atype = None, anum = None, mass = None, charge = None, ptype = None, sigma = None, epsilon = None):
        self.atype = atype
        self.anum = anum
        self.mass = mass
        self.charge = charge
        self.ptype = ptype
        self.sigma = sigma
        self.epsilon = epsilon            

class bondType:
    def __init__(self, atypes=None, func=None, length=None, fconstant=None):
        self.atypes=atypes
        self.func=func
        self.length=length
        self.fconstant=fconstant
    
class angleType:
    def __init__ (self, atypes=None, func=None, angle=None, fconstant=None, ubval=None, ubfconstant=None):
        self.atypes = atypes
        self.func = func
        self.angle = angle
        self.fconstant = fconstant
        self.ubval = ubval
        self.ubfconstant = ubfconstant
    
class dihedralType:
    def __init__ (self, atypes = None, func = None, angle = None, fconstant = None, mult = None):
        self.atypes = atypes
        self.func = func
        self.angle = angle
        self.fconstant = fconstant
        self.mult = mult

class improperType:
    def __init__ (self, atypes = None, func = None, angle = None, fconstant = None):
        self.atypes = atypes
        self.func = func
        self.angle = angle
        self.fconstant = fconstant
    
class pairType:
    def __init__(self, atypes = None, func = None, sigma = None, epsilon = None):
        self.atypes = atypes
        self.func = func
        self.sigma = sigma
        self.epsilon = epsilon

class ffBonded:  
    """Class to store all the bonded interactions in FF"""
    def __init__ (self, bonds = newDict(), angles = newDict(), dihedrals = newDict(), impropers = newDict()):
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.impropers = impropers
        
class ffNonBonded:
    """Class to store all the non-bonded interactions in FF"""
    def __init__(self, atoms = {}, pairs = newDict()):
        self.atoms = atoms
        self.pairs = pairs
                                

class ffCommon:
    """Simply unites Bonded and NonBonded classes"""
    def __init__(self, Bonded = ffBonded(), NonBonded = ffNonBonded(), defdict = {}):
        self.Bonded = Bonded
        self.NonBonded = NonBonded
        self.defdict = defdict
