
import math
import numpy as np
from numpy.linalg import norm
from itertools import combinations, permutations
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem 
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from intparams import *
from restop import *


# In[ ]:


def rtpreader(*rtpfiles): 
    """Reads rtp files; returns a residue dict in {ResName: (resType obj)} format"""
    
    mode = ''
    resname = 'EMPTY'
    btypeslist = ['bonds','angles','dihedrals','impropers','all_dihedrals','nrexcl','HH14','RemoveDih']
    modeslist = ['bondedtypes', 'atoms', 'bonds', 'angles', 'dihedrals', 'impropers']
    errorlist = []
    sline = []
    atomnums = {}   # Name: number
    ff_atypes = {}    # Name: FF type
    ff_atypes_com = {} # Res: dict of name-fftype
    btypes = {}
    i = 0
    resdict = {}
    rtpdefangles = []
    rtpdefdihedrals = []
    rtpdefimps = []
    for rtpfile in rtpfiles:
        with open(rtpfile) as f:     # Opening .rtp file
            mol = Chem.MolFromSmiles('')     # Getting an empty molecule and making it editable:
            mw = Chem.EditableMol(mol)
            for line in f:
                if line.strip() and line.split()[0][0]!=';' :     # Checking if the line is not empty or commented
                    sline = line.split()     # Splitting the line to its elements
                    if sline[0] == '[':     # If the line is a header --
                        if sline[1] in modeslist: # and it could be a mode --
                            mode = sline[1]         # it's a mode.
                        else:                               #If it can't be a mode, it's a residue name
                            resdict[resname] = resType(resname, mw.GetMol(), ff_atypes, rtpdefangles, rtpdefdihedrals, rtpdefimps) # Adding the mol to res dict
                            #resdict[resname].ff = ff
                            ff_atypes_com[resname]=ff_atypes           # Adding atypes dict to common dict
                            resname = sline[1]                        # Changing the name
                            mol = Chem.MolFromSmiles('')              # Resetting mol & dicts & mode & counter
                            mw = Chem.EditableMol(mol)
                            atomnums = {}
                            ff_atypes = {} 
                            mode = ''
                            i = 0
                            rtpdefangles = []
                            rtpdefdihedrals = []
                            rtpdefimps = []
                    else:                   # Else, if the line has information:
                        if mode == 'bondedtypes':
                            for n, item in enumerate(btypeslist):
                                btypes[item]=sline[n]         
                        elif mode == 'atoms':     
                            name = sline[0]      # Getting the name of atom
                            atype = GetAtomType(sline[1])      # Getting the element
                            ff_atypes[name]=sline[1]    # Dict: what type is an atom with this name
                            if atype!="*":
                                mw = AddNamedAtom(resname, name, atype, mw)
                                atomnums[name]=i                # Dict: what number in RDKit is an atom with this name
                                i+=1
                        elif mode == 'bonds':
                            name1 = sline[0]      # Getting names of bonded atoms
                            name2 = sline[1]
                            if name1[0]!='-' and name1[0]!='+' and name2[0]!='-' and name2[0]!='+':
                                if GetAtomType(ff_atypes[name1])!='*' and GetAtomType(ff_atypes[name2])!='*':
                                    n1 = atomnums[name1]     # Getting their numbers from dict
                                    n2 = atomnums[name2]
                                    try: 
                                        mw.AddBond(n1, n2, order = Chem.rdchem.BondType.SINGLE)
                                    except RuntimeError:
                                        errorlist.append((name1, name2))
                        elif mode == 'angles':
                            rtpdefangles.append((sline[0], sline[1], sline[2]))
                        elif mode == 'dihedrals':
                            rtpdefdihedrals.append((sline[0], sline[1], sline[2], sline[3]))
                        elif mode == 'impropers':
                            rtpdefimps.append((sline[0], sline[1], sline[2], sline[3]))
    resdict[resname] = resType(resname, mw.GetMol(), ff_atypes, rtpdefangles, rtpdefdihedrals, rtpdefimps)
    resname = sline[1]    
    del resdict['EMPTY']
    
    if not resdict:
        print('No molecule found')
        
    return resdict     


# In[ ]:


def itpreader(*itpfiles):
    bonds = newDict()
    angles = newDict()
    dihedrals = newDict()
    impropers = newDict()
    pairs = newDict()
    mode = ''
    for itpfile in itpfiles: 
        with open (itpfile) as f:
            for line in f:
                if line.strip() and line.split()[0][0]!=';':
                    sline = line.split()
                    if sline[0]=='[':
                        mode = sline[1]
                    else:
                        if mode == 'bondtypes':
                            bond = bondType(atypes=sline[0:2],func=sline[2],length=sline[3],fconstant=sline[4])
                            bonds[bond.atypes] = bond

                        elif mode == 'angletypes':
                            angle = angleType(atypes=sline[0:3], func=sline[3], angle=sline[4], fconstant=sline[5],                                               ubval=sline[6], ubfconstant=sline[7])
                            angles[angle.atypes] = angle

                        elif mode == 'dihedraltypes':
                            if sline[4]=='2' or sline[4]=='4':
                                improper = improperType(atypes = sline[0:4], func = sline[4], angle = sline[5],                                                         fconstant = sline[6])
                                impropers[improper.atypes] = improper

                            else:
                                dihedral = dihedralType(atypes = sline[0:4], func = sline[4], angle = sline[5],                                                     fconstant = sline[6], mult = sline[7])
                                dihedrals[dihedral.atypes] = dihedral
                        elif mode == 'pairtypes':
                            pair = pairType(sline[0:2], sline[2], sline[3], sline[4])
                            pairs[pair.atypes] = pair
                            
    ff = ffCommon(ffBonded(bonds, angles, dihedrals, impropers), ffNonBonded(pairs))
    return ff

