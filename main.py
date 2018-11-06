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


def rtpreader(*rtpfiles, ff = ffCommon()):   # NEEDS A LOT OF EXTRA FEATURES
    """Reads rtp files; returns a residue dict in {ResName: (resType obj)} format"""
    
    mode = ''
    resname = 'EMPTY'
    btypeslist = ['bonds','angles','dihedrals','impropers','all_dihedrals','nrexcl','HH14','RemoveDih']
    modeslist = ['bondedtypes', 'atoms', 'bonds', 'angles', 'dihedrals', 'impropers']
    errorlist = []
    sline = []
    atomnums = {}   # Name: number
    ff_atypes = {}    # Name: FF type
    atomcharges = {}
    btypes = {}
    i = 0
    resdict = {}
    rtpdefatoms = []
    rtpdefbonds = []
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
                            resdict[resname] = resType(resname, mw.GetMol(), ff_atypes, atomcharges, rtpdefatoms, rtpdefbonds, rtpdefangles, rtpdefdihedrals, rtpdefimps) # Adding the mol to res dict
                            resname = sline[1]                        # Changing the name
                            mol = Chem.MolFromSmiles('')              # Resetting mol & dicts & mode & counter
                            mw = Chem.EditableMol(mol)
                            atomnums = {}
                            ff_atypes = {} 
                            atomcharges = {}
                            mode = ''
                            i = 0
                            rtpdefbonds = []
                            rtpdefangles = []
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
                            if len(sline)>2:   # If something more than atomname-atomtype is defined
                                rtpdefatoms.append([atype, sline[2:]]
                                try:
                                    atomcharges[i]=Decimal(sline[3])
                                except:
                                    atomcharges[i]=None
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
                                        errorlist.append(resname, (name1, name2))
                            if len(sline)>2:
                                atypes = (name1, name2)
                                rtpdefbonds.append([atypes, sline[2:]])
                        elif mode == 'angles':
                            atypes = (sline[0], sline[1], sline[2])
                            if len(sline)>3:
                                rtpdefangles.append([atypes, sline[3:]])
                        elif mode == 'dihedrals':
                            atypes = (sline[0], sline[1], sline[2], sline[3])
                            if len(sline)>4:
                                rtpdefdihedrals.append([atypes, sline[4:]])
                        elif mode == 'impropers':
                            atypes = (sline[0], sline[1], sline[2], sline[3])
                            if len(sline)>4:
                                rtpdefimps.append([atypes, sline[4:]])
    resdict[resname] = resType(resname, mw.GetMol(), ff_atypes, rtpdefatoms, rtpdefbonds, rtpdefangles, rtpdefdihedrals, rtpdefimps)
    del resdict['EMPTY']
    if errorlist:
        print('Following bonds caused a RuntimeError: ', errorlist)
    if not resdict:
        print('No molecule found')
    if len(resdict)==1:
        return resdict[resname]
    else:    
        return resdict     


def itpreader(*itpfiles):
    atoms = {}
    bonds = newDict()
    angles = newDict()
    dihedrals = newDict()
    impropers = newDict()
    pairs = newDict()
    defdict = {}
    mode = ''
    for itpfile in itpfiles: 
        with open (itpfile) as f:
            ifcounter = 0
            for line in f:
                if line.strip() and line.split()[0][0]!=';':
                    sline = line.split()
                    if sline[0]=='#define':
                        defdict[sline[1]] = sline[2:]
                    if sline[0]=='#ifdef':
                        ifcounter+=1
                    if ifcounter==0:
                        if sline[0]=='[':
                            mode = sline[1]
                        else:
                            if mode == 'atomtypes': 
                                try:
                                    atom = atomType(atype = sline[0], anum = sline[1], mass = sline[2], charge = sline[3], ptype = sline[4], sigma = float(sline[5]), epsilon = float(sline[6]))
                                    atoms[atom.atype] = atom
                                except IndexError:
                                    print('Following atom lacks parameters: '+sline[0]'. It was not added to the parametrized force field.')
                            elif mode == 'bondtypes':
                                try:
                                    bond = bondType(atypes=sline[0:2],func=sline[2],length=sline[3],fconstant=sline[4])
                                    bonds[bond.atypes] = bond
                                except IndexError:
                                    print('Following bond lacks parameters: '+sline[0:2]'. It was not added to the parametrized force field.')

                            elif mode == 'angletypes':
                                try:
                                    angle = angleType(atypes=sline[0:3], func=sline[3], angle=sline[4], fconstant=sline[5],                                               ubval=sline[6], ubfconstant=sline[7])
                                   angles[angle.atypes] = angle
                                except IndexError:
                                    print('Following angle lacks parameters: '+sline[0:3]'. It was not added to the parametrized force field.')

                            elif mode == 'dihedraltypes':
                                try:
                                    f = sline[4]
                                    if sline[4]=='2' or sline[4]=='4':
                                        try:
                                            improper = improperType(atypes = sline[0:4], func = sline[4], angle = sline[5],                                                         fconstant = sline[6])
                                            impropers[improper.atypes] = improper
                                        except IndexError:
                                            print('Following dihedral lacks parameters: '+sline[0:4]'. It was not added to the parametrized force field.')

                                    else:
                                        try:
                                            dihedral = dihedralType(atypes = sline[0:4], func = sline[4], angle = sline[5],                                                     fconstant = sline[6], mult = sline[7])
                                            dihedrals[dihedral.atypes] = dihedral
                                        except IndexError:
                                            print('Following improper lacks parameters: '+sline[0:4]'. It was not added to the parametrized force field.')
                               except IndexError:
                                    print('Following dihedral lacks parameters: '+sline[0:4]'. It was not added to the parametrized force field.')
                            elif mode == 'pairtypes':
                                try:
                                    pair = pairType(sline[0:2], sline[2], sline[3], sline[4])
                                    pairs[pair.atypes] = pair
                                except IndexError:
                                    print('Following bond lacks parameters: '+sline[0:2]'. It was not added to the parametrized force field.')
                    if sline[0] == '#endif':
                        ifcounter-=1        
    ff = ffCommon(ffBonded(bonds, angles, dihedrals, impropers), ffNonBonded(atoms, pairs), defdict)
    return ff

