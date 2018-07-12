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


def GetAtomName(atom):
    """Returns RDKit atom name from PDB Residue info"""
    pdbinf = atom.GetPDBResidueInfo()
    name = pdbinf.GetName()
    return name.strip()

def GetAtomType(ortype):
    """Returns atom element out of its PDB type"""
    if ortype[0]=='-' or ortype[0]=='+':
        atype = ortype[1:]
    else:
        atype = ortype
    if atype[:2]=='CL':
        atom = 'Cl'
    elif atype[:2]=='BR':
        atom = 'Br'
    elif atype[:2]=='AG':
        atom = 'Ag'
    elif atype[:2]=='AL':
        atom = 'Al'
    elif atype[:2]=='AU':
        atom = 'Au'
    elif atype[:2]=='MG':
        atom = 'Mg'
    elif atype[:2]=='ZN':
        atom = 'Zn'
    elif atype[:3]=='RUB':
        atom = 'Rb'
    elif atype[0]=='L' or atype[:3]=='DUM':
        atom = '*'
    else:
        atom = atype[0]
    return atom

def AddNamedAtom(resname, atomname, atomelement, edmol):
    """Adds a RDKit atom type to an editable molecule with PDB residue info by creating a temporary pdb file"""
    with open ('temp.pdb', 'w') as temp: 
        string = 'ATOM      1 %4s %6s  1       0.000   0.000   0.000  1.00  1.00          %2s'%(atomname, resname, atomelement)
        temp.write (string) 
    m = Chem.MolFromPDBFile('temp.pdb')     # Creating an one-atom molecule 
    edmol.AddAtom(m.GetAtomWithIdx(0))     # Adding its only atom to editable system
    return edmol


def atomanglewedge(x, y, ax, t):    
    xa = x[0]
    ya = y[0]
    xo = x[1]
    yo = y[1]
    xb = x[2]
    yb = y[2]
    hor = [1,0]                      # Horizontal vector, since angles are measured from that
    oa = [xa-xo, ya-yo]             # First vector coordinates
    ob = [xb-xo, yb-yo]             # Second vector coordinates
    r = norm(oa)/t              # Radius of wedge
    a_cos =  np.dot(oa, hor)/(norm(oa)*norm(hor))        # Cosines from scalar multiplication
    b_cos = np.dot(ob, hor)/(norm(ob)*norm(hor))

    if oa[1]<0:                                    # In case vector is directed down and angle is not the one we want
        a_angle = 2*np.pi - math.acos(a_cos)
    else:
        a_angle = math.acos(a_cos)
    if ob[1]<0:
        b_angle = 2*np.pi - math.acos(b_cos)
    else:
        b_angle = math.acos(b_cos)
    theta1 = a_angle*180/np.pi 
    theta2 = b_angle*180/np.pi 
    if (a_angle>b_angle and (a_angle-b_angle)<np.pi) or (b_angle>a_angle and (b_angle-a_angle)>np.pi):
        theta1 = b_angle*180/np.pi 
        theta2 = a_angle*180/np.pi 

    tmp = patches.Wedge([xo,yo], theta1 = theta1, theta2 = theta2, color='r', r=r, alpha = 0.2)
    ax.add_artist(tmp)


def bezier(xlist, ylist, ax, color):
    # Gets list of x coords, list of y coords & editable molecule
    k = 0.9
    gx1 = xlist[0]
    gx2 = xlist[1]
    gx3 = xlist[2]
    gx4 = xlist[3]
    gy1 = ylist[0]
    gy2 = ylist[1]
    gy3 = ylist[2]
    gy4 = ylist[3]
    x0 = gx1
    y0 = gy1
    cx0 = gx2
    cy0 = gy2
    cx1 = gx3
    cy1 = gy3
    x1 = gx4
    y1 = gy4
    # Vector making
    ab = [cx0-x0, cy0-y0]
    bc = [cx0-cx1, cy0-cy1]
    cd = [x1-cx1, y1-cy1]
    # Case they're almost on the same line -- moving the control point perpendicularly to them
    if np.abs(np.dot(ab, bc)/(norm(ab)*norm(bc)))>=0.95:
        dv = [cy0-y0, x0-cx0]/norm([cy0-y0, x0-cx0])
        cx0+=dv[0]*k
        cy0+=dv[1]*k
    if np.abs(np.dot(bc, cd)/(norm(bc)*norm(cd)))>=0.95:
        dv = [cy0-cy1, cx1-cx0]/norm([cy0-cy1, cx1-cx0])
        cx1+=dv[0]*k
        cy1+=dv[1]*k
    xb = [gx4, gx3, gx2, gx1]
    yb = [gy4, gy3, gy2, gy1]
    for s in range(0, 101, 2):
        t = s/100
        a = ((1-t)**3)*x0 + 3*((1-t)**2)*t*cx0 +3*(1-t)*(t**2)*cx1 + (t**3)*x1 
        b =  ((1-t)**3)*y0 + 3*((1-t)**2)*t*cy0 +3*(1-t)*(t**2)*cy1 + (t**3)*y1 
        xb.append(a)
        yb.append(b)
    ax.fill(xb, yb, alpha = 0.4, color = color)
    return ax


# In[4]:


class resType:    
    """Class to keep and work with all the residues in. \n     Includes: .name -- a residue name, .mol -- corresponding mol obj, .ff -- ffCommon obj, atomtypes, atomnums,     numtypes -- dicts, rtpdefimps -- impropers defined in rtp file, itpfoundimps -- impropers from itp file     that exist in this molecule"""
    def __init__(self, name = '', mol = Chem.MolFromSmiles(''), atomtypes = {}, rtpdefangles = [], rtpdefdihedrals = [], rtpdefimps = [], ff=ffCommon()):
        self.name = name
        self.error = None
        self.mol = mol
        self.atomtypes = atomtypes
        self.rtpdefangles = rtpdefangles
        self.rtpdefdihedrals = rtpdefdihedrals
        self.rtpdefimps = rtpdefimps
        self.extraff = None    # Not parametrized FF
        self.itpfoundimps = []
        self.atomnums = {}
        for atom in self.mol.GetAtoms():
            aidx = atom.GetIdx()
            aname = GetAtomName(atom)
            self.atomnums[aname] = aidx
        self.numtypes = {}
        for key in self.atomnums.keys():
            anum = self.atomnums[key]
            atype = self.atomtypes[key]
            self.numtypes[anum] = atype
            
        self.ff = ff       # Parametrized FF    
        
    def WriteExtraFF(self):
        self.extraff = ffCommon()
        self.xbonds = newDict()
        self.xangles = newDict()
        self.xdihedrals = newDict()
        self.ximpropers = newDict()
        resbonds = self.CheckBonds()
        if resbonds:
            for resbond in resbonds:
                if not self.xbonds[resbond]:
                    self.xbonds[resbond] = bondType(atypes = resbond)
        resangles = self.CheckAngles()
        if resangles:
            for resangle in resangles: 
                if not self.xangles[resangle]:
                    self.xangles[resangle] = angleType(atypes = resangle)
        resdihedrals = self.CheckDihedrals()
        if resdihedrals:
            for resdihedral in resdihedrals:
                if not self.xdihedrals[resdihedral]:
                    self.xdihedrals[resdihedral] = dihedralType(atypes = resdihedral)
        resimps = self.CheckImpropers()
        if resimps:
            for resimp in resimps:
                if not self.ximpropers[resimp]:
                    self.ximpropers[resimp] = improperType(atypes = resimp)
        self.extraff.Bonded.bonds = self.xbonds
        self.extraff.Bonded.angles = self.xangles
        self.extraff.Bonded.dihedrals = self.xdihedrals
        self.extraff.Bonded.impropers = self.ximpropers
    
    def AddAttachedAtom(self, atomname='', atomelement='', atomtype = '', idx2bond=0, order = 'single'):
        """Adds a RDKit atom type to a res molecule with PDB residue info and bonds it with desirable"""
        orderdict = {'single': Chem.rdchem.BondType.SINGLE, 'double': Chem.rdchem.BondType.DOUBLE, 'triple': Chem.rdchem.BondType.TRIPLE, 'aromatic': Chem.rdchem.BondType.AROMATIC}
        if atomname in self.atomtypes:
            write['Name already taken']
        else:
            idx = self.mol.GetNumAtoms()
            edmol = Chem.EditableMol(self.mol)
            AddNamedAtom(self.name, atomname, atomelement, edmol)
            edmol.AddBond(idx2bond, idx, order = orderdict[order])
            m = edmol.GetMol()
            Chem.SanitizeMol(m)
            self.atomtypes[atomname] = atomtype
            self.numtypes = {}
            self.atomnums = {}
            for atom in m.GetAtoms():
                ai = atom.GetIdx()
                aname = GetAtomName(atom)
                self.atomnums[aname] = ai
                atype = self.atomtypes[aname]
                self.numtypes[ai] = atype
            self.mol = m
        
    def RemoveAtom(self, atomidx):
        atom = self.mol.GetAtomWithIdx(atomidx)
        name = GetAtomName(atom)
        edmol = Chem.EditableMol(self.mol)
        edmol.RemoveAtom(atomidx)
        m = edmol.GetMol()
        Chem.SanitizeMol(m)
        self.mol = m
        del self.atomtypes[name]
        self.atomnums = {}
        for atom in self.mol.GetAtoms():
            aidx = atom.GetIdx()
            aname = GetAtomName(atom)
            self.atomnums[aname] = aidx
        self.numtypes = {}
        for key in self.atomnums.keys():
            anum = self.atomnums[key]
            atype = self.atomtypes[key]
            self.numtypes[anum] = atype
        for atom in m.GetAtoms():
            ai = atom.GetIdx()
            aname = GetAtomName(atom)
            if self.atomnums[aname]!=ai:
                print(ai)
        
    def FindMolAngles(self):
        """Returns a list of all angles in the molecule"""
        angles = []
        seen = set()
        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            for neighbor in a2.GetNeighbors():
                if neighbor.GetIdx()!=a1.GetIdx():
                    angle = (GetAtomName(a1), GetAtomName(a2), GetAtomName(neighbor))
                    if angle not in seen and tuple(reversed(angle)) not in seen:
                        seen.add(angle)
                        angles.append(angle)
        return angles  

    def FindMolDihedrals(self):
        """Returns a list of all dihedrals in the molecule"""
        dihedrals = []
        seen = set()
        mol = self.mol
        for bond in mol.GetBonds():
            a2 = bond.GetBeginAtom()
            a3 = bond.GetEndAtom()
            for a1 in a2.GetNeighbors():
                if a1.GetIdx()!= a3.GetIdx():
                    for a4 in a3.GetNeighbors():
                        if a4.GetIdx()!= a2.GetIdx():
                            dihedral = (GetAtomName(a1), GetAtomName(a2), GetAtomName(a3),GetAtomName(a4))
                            if dihedral not in seen and tuple(reversed(dihedral)) not in seen:
                                seen.add(dihedral)
                                dihedrals.append(dihedral)
        return dihedrals    

    def GetBond(self, aidx1 = -1, aidx2 = -1, aname1 = '', aname2 = ''):
        """Returns a bondType obj corresponding to atoms with given idxs or names"""
        if (aidx1+aidx2)<0:
            atype1 = self.atomtypes[aname1]
            atype2 = self.atomtypes[aname2]
        else:
            atype1 = self.numtypes[aidx1]
            atype2 = self.numtypes[aidx2]
        bond = self.ff.Bonded.bonds[atype1, atype2]
        if bond == None:
            print("Bond doesn't exist in given force field: "+atype1 + ' '+atype2)
        return bond
    
    def GetPair(self, aidx1 = -1, aidx2 = -1, aname1 = '', aname2 = ''):
        """Returns a pairType obj corresponding to atoms with given idxs or names"""
        if (aidx1+aidx2)<0:
            atype1 = self.atomtypes[aname1]
            atype2 = self.atomtypes[aname2]
        else:
            atype1 = self.numtypes[aidx1]
            atype2 = self.numtypes[aidx2]
        pair = self.ff.NonBonded.pairs[atype1, atype2]
        if pair == None:
            print("Pair doesn't exist in given force field: "+atype1 + ' '+atype2)
        return pair
        
    def GetAngle(self, aidx1 = -1, aidx2 = -1, aidx3 = -1):
        """Returns an angleType obj corresponding to atoms with given idxs"""
        atype1 = self.numtypes[aidx1]
        atype2 = self.numtypes[aidx2]
        atype3 = self.numtypes[aidx3]
        angle = self.ff.Bonded.angles[atype1, atype2, atype3]
        if angle == None:
            print ("Angle doesn't exist in given force field: " + atype1 + ' ' + atype2 + ' ' + atype3)
        return angle
    
    def GetDihedral(self, aidx1 = -1, aidx2 = -1, aidx3 = -1, aidx4 = -1):
        """Returns a dihedralType obj corresponding to atoms with given idxs"""
        atype1 = self.numtypes[aidx1]
        atype2 = self.numtypes[aidx2]
        atype3 = self.numtypes[aidx3]
        atype4 = self.numtypes[aidx4]
        dihedral = self.ff.Bonded.dihedrals[atype1, atype2, atype3, atype4]
        atypes = (atype1, atype2, atype3, atype4)
        if dihedral == None:
            k = 0
            for i in range(4):
                alist = list(atypes)
                alist[i]='X'
                blist = alist[:]
                for j in range(4):
                    if j>i:
                        alist[j]='X'
                    if self.ff.Bonded.dihedrals[tuple(alist)]:
                        k+=1
                        dihedral = self.ff.Bonded.dihedrals[tuple(alist)]
                    alist = blist[:]
            if k==0:
                print ("Dihedral doesn't exist in given force field: " + atype1 + ' ' + atype2 + ' ' + atype3 + ' '                       + atype4)
        return dihedral
    
    def GetImproper(self, aidx1 = -1, aidx2 = -1, aidx3 = -1, aidx4 = -1):
        """Returns a improperType obj corresponding to atoms with given idxs"""
        atype1 = self.numtypes[aidx1]
        atype2 = self.numtypes[aidx2]
        atype3 = self.numtypes[aidx3]
        atype4 = self.numtypes[aidx4]
        improper = self.ff.Bonded.impropers[atype1, atype2, atype3, atype4]
        atypes = (atype1, atype2, atype3, atype4)
        if improper == None:
            k = 0
            for i in range(4):
                alist = list(atypes)
                alist[i]='X'
                blist = alist[:]
                for j in range(4):
                    if j>i:
                        alist[j]='X'
                    if self.ff.Bonded.impropers[tuple(alist)]:
                        k+=1
                        improper = self.ff.Bonded.impropers[tuple(alist)]
                    alist = blist[:]
            if k==0:
                print ("Improper doesn't exist in given force field: " + atype1 + ' ' + atype2 + ' ' + atype3 + ' '                       + atype4)
        return improper
    
    def CheckBonds(self):
        """Checks mol bonds for parametrization, returns atom types"""
        mol = self.mol
        missingbonds = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            a1type = self.numtypes[a1]
            a2type = self.numtypes[a2]
            atypes = (a1type, a2type)
            if not self.ff.Bonded.bonds[atypes]:
                missingbonds.append((a1type,a2type))
        return missingbonds  
    
    def CheckAngles(self):
        """Checks mol angles for parametrization, returns atom types"""
        mol = self.mol
        missingangles = []
        molangles = self.FindMolAngles()
        for molangle in molangles:
            a1type = self.atomtypes[molangle[0]]
            a2type = self.atomtypes[molangle[1]]
            a3type = self.atomtypes[molangle[2]]
            atypes = (a1type, a2type, a3type)
            if not self.ff.Bonded.angles[atypes]:
                missingangles.append((a1type,a2type, a3type))
        return missingangles
    
    def CheckDihedrals(self):
        """Checks mol dihedral for parametrization, returns atom types"""
        mol = self.mol
        missingdihedrals = []
        moldihedrals = self.FindMolDihedrals()
        for moldihedral in moldihedrals:
            a1type = self.atomtypes[moldihedral[0]]
            a2type = self.atomtypes[moldihedral[1]]
            a3type = self.atomtypes[moldihedral[2]]
            a4type = self.atomtypes[moldihedral[3]]
            atypes = (a1type, a2type, a3type, a4type)
            if not self.ff.Bonded.dihedrals[atypes]:
                k = 0
                for i in range(4):
                    alist = list(atypes)
                    alist[i]='X'
                    blist = alist[:]
                    for j in range(4):
                        if j>i:
                            alist[j]='X'
                        if self.ff.Bonded.dihedrals[tuple(alist)]:
                            k+=1
                        alist = blist[:]
                if k==0:
                    missingdihedrals.append(atypes)
        return missingdihedrals
    
    def CheckImpropers(self):
        """Checks rtp-defined impropers for parametrization, returns atom types"""
        missingimpropers = []
        rtpimps = self.rtpdefimps
        if not rtpimps:
            print ('No impropers defined in .rtp file')
        else:
            for imp in rtpimps:
                a1type = self.atomtypes[imp[0]]
                a2type = self.atomtypes[imp[1]]
                a3type = self.atomtypes[imp[2]]
                a4type = self.atomtypes[imp[3]]
                atypes = (a1type, a2type, a3type, a4type)
                if not self.ff.Bonded.impropers[atypes]:
                    k = 0
                    for i in range(4):
                        alist = list(atypes)
                        alist[i]='X'
                        blist = alist[:]
                        for j in range(4):
                            if j>i:
                                alist[j]='X'
                            if self.ff.Bonded.impropers[tuple(alist)]:
                                k+=1
                            alist = blist[:]
                    if k==0:
                        missingimpropers.append(atypes)
            return missingimpropers
    
    

    def FindItpImpropers(self):
        """Checks mol object for impropers defined by atom types in itp files"""
        mol = self.mol
        nametypes = self.atomtypes
        imps = []
        linearimps = []
        roundimps = []
        seen = set()
        for bond in mol.GetBonds():
            a2 = bond.GetBeginAtom()  # Long story short, it's a1        a4
            a3 = bond.GetEndAtom()                      #       \       /
            for a1 in a2.GetNeighbors():                     #   a2 — a3
                if a1.GetIdx()!= a3.GetIdx():
                    for a4 in a3.GetNeighbors():
                        if a4.GetIdx()!= a2.GetIdx():  
                            impnames = (GetAtomName(a1), GetAtomName(a2), GetAtomName(a3),                                        GetAtomName(a4)) #Names of said atoms
                            if impnames not in seen and tuple(reversed(impnames)) not in seen:
                                seen.add(impnames)
                                imptypes = (nametypes[impnames[0]], nametypes[impnames[1]], nametypes[impnames[2]],                                             nametypes[impnames[3]])   # Getting tuple of their types
                                if self.ff.Bonded.impropers[imptypes]:      # Checking if it's in our ff dictionary
                                    linearimps.append(impnames)
                                else:                            #  Checking for Xs
                                    ki = 0
                                    for ii in range(4):
                                        alist = list(imptypes)
                                        alist[ii]='X'
                                        blist = alist[:]
                                        for ji in range(4):
                                            if ji>ii:
                                                alist[ji]='X'
                                            if self.ff.Bonded.impropers[tuple(alist)]:
                                                ki+=1
                                            alist = blist[:]
                                    if ki>0:
                                        linearimps.append(impnames)

        for atom in mol.GetAtoms():
            iname = [GetAtomName(atom)]
            neighbornames = []
            for neighbor in atom.GetNeighbors():  # Getting aaaaall the neighbors
                neighbornames.append(GetAtomName(neighbor))         # And writing their names down
            if len(neighbornames)>2:                              # If there are 3 or more, we can work with it
                namecombs = list(permutations(neighbornames, 3))   # All permutations of every combination of 3 neighbor
                for namecomb in namecombs:                     # Iterating through them
                    impnames = iname + list(namecomb)            # Let's pretend there is order & i-atom should go first
                    if tuple(impnames) not in seen:            # Checking in seen oh yeah
                        seen.add(tuple(impnames))           # If it's new, putting it in 'seen'
                        imptypes = (nametypes[impnames[0]], nametypes[impnames[1]], nametypes[impnames[2]],                                     nametypes[impnames[3]]) # Getting types from our little name-type dict
                        if self.ff.Bonded.impropers[imptypes]:  # Well you know whats going on here youve seen it before
                            roundimps.append(tuple(impnames))
                        else:
                            ki = 0
                            for ii in range(4):
                                alist = list(imptypes)
                                alist[ii]='X'
                                blist = alist[:]
                                for ji in range(4): 
                                    if ji>ii:
                                        alist[ji]='X'
                                    if self.ff.Bonded.impropers[tuple(alist)]:
                                        ki+=1
                                    alist = blist[:]
                            if ki>0:
                                roundimps.append(tuple(impnames))
        self.itpfoundimps = linearimps[:]+roundimps[:]
        self.itpfoundlinearimps = linearimps[:]
        self.itpfoundroundimps = roundimps[:]
            
    def SetBondParams(self, bond, f=None, b0=None, kb=None):
        """Sets some bondType parameter in non-parametrized force field"""
        if not self.extraff:
            print("No unparametrized ff detected! Either you're good, or haven't called 'res.WriteExtraFF()' function. Try the latter.")
        else:
            if f:
                self.extraff.Bonded.bonds[bond].func = f
            if b0:
                self.extraff.Bonded.bonds[bond].length = b0
            if kb:
                self.extraff.Bonded.bonds[bond].fconstant = kb
            
    def SetAngleParams(self, angle, f=None, theta0=None, ktheha=None, ub0=None, kub=None):
        """Sets some angleType parameter in non-parametrized force field"""
        if not self.extraff:
            print("No unparametrized ff detected! Either you're good, or haven't called 'res.WriteExtraFF()' function. Try the latter.")
        else:
            if f:
                self.extraff.Bonded.angles[angle].func = f
            if theta0:
                self.extraff.Bonded.angles[angle].angle = theta0
            if ktheha:
                self.extraff.Bonded.angles[angle].fconstant = ktheha
            if ub0:
                self.extraff.Bonded.angles[angle].ubval = ub0
            if kub:
                self.extraff.Bonded.angles[angle].ubfconstant = kub
            
    def SetDihedralParams(self, dihedral, f = None, phi0=None, kphi=None, mult=None):
        """Sets some dihedralType parameter in non-parametrized force field"""
        if not self.extraff:
            print("No unparametrized ff detected! Either you're good, or haven't called 'res.WriteExtraFF()' function. Try the latter.")
        else:
            if f:
                self.extraff.Bonded.dihedrals[dihedral].func = f
            if phi0:
                self.extraff.Bonded.dihedrals[dihedral].angle = phi0
            if kphi:
                self.extraff.Bonded.dihedrals[dihedral].fconstant = kphi
            if mult:
                self.extraff.Bonded.dihedrals[dihedral].mult = mult
            
    def SetImproperParams(self, dihedral, f = None, phi0=None, kphi=None):
        """Sets some improperType parameter in non-parametrized force field"""
        if not self.extraff:
            print("No unparametrized ff detected! Either you're good, or haven't called 'res.WriteExtraFF()' function. Try the latter.")
        else:
            if f:
                self.extraff.Bonded.impropers[improper].func = f
            if phi0:
                self.extraff.Bonded.impropers[improper].angle = phi0
            if kphi:
                self.extraff.Bonded.impropers[improper].fconstant = kphi
            if mult:
                self.extraff.Bonded.dihedrals[dihedral].mult = mult
    
    def WriteSuppBonded(self, fname='newitp.itp'):  
        """Writes a supporting itp file with not-parametrized bonds, angles, dihedrals, impropers (if defined in rtp)"""
        if self.extraff:
            with open(fname, 'w') as f:
                f.write('[ bondtypes ]\n')
                f.write(';'+ '%6s %7s %5s %12s %12s \n' %('i','j','func','b0','kb'))
                for bond in self.extraff.Bonded.bonds:
                    btype = self.extraff.Bonded.bonds[bond]
                    f.write('%7s %7s' %bond+' %5s %12s %12s'%(btype.func, btype.length,btype .fconstant)+'\n')
                f.write('\n [ angletypes ] \n')
                f.write(';' + '%7s %8s %8s %5s %12s %12s %12s %12s \n' %('i','j','k','func','theta0','ktheha','ub0','kub'))
                for angle in  self.extraff.Bonded.angles:
                    f.write('%8s %8s %8s' %angle + '\n')
                f.write('\n [ dihedraltypes ] \n')
                f.write(';'+'%7s %8s %8s %8s %5s %12s %12s %5s \n' %('i','j','k','l','func','phi0','kphi','mult'))
                for dihedral in  self.extraff.Bonded.dihedrals:
                    f.write('%8s %8s %8s %8s' %dihedral + '\n')
                f.write('\n [ dihedraltypes ] \n')
                f.write("; 'improper' dihedrals \n")
                f.write(';'+'%7s %8s %8s %8s %5s %12s %12s \n' %('i','j','k','l','func','phi0','kphi'))
                for improper in  self.extraff.Bonded.impropers:
                    f.write('%8s %8s %8s %8s' %improper + '\n')
        else:
            print("No unparametrized ff detected! Either you're good, or haven't called 'res.WriteExtraFF()' function. Try the latter.") 
                    
    def WriteRtp(self, fname = 'newrtp.rtp'):
        with open(fname, 'w') as f:
            f.write('[ '+self.name+' ] \n')
            f.write(' [ atoms ] \n')
            seen = set()
            for atom in self.mol.GetAtoms():
                pdbinf = atom.GetPDBResidueInfo()
                aname = pdbinf.GetName()
                if aname in seen:
                    print(aname, atom.GetIdx())
                else:
                    seen.add(aname)
                if aname.strip() in self.atomtypes:
                    atype = self.atomtypes[aname.strip()]
                else:
                    atype = '     '
                f.write('%9s %8s' %(aname, atype) + '\n')

            f.write('\n')
            f.write(' [ bonds ] \n')
            for bond in self.mol.GetBonds():
                batom = bond.GetBeginAtom()
                eatom = bond.GetEndAtom()
                bname = GetAtomName(batom)
                ename = GetAtomName(eatom)
                f.write('%9s %6s' %(bname, ename) + '\n')
            if self.rtpdefangles:
                f.write('\n')
                f.write(' [ angles ] \n')
                for angle in self.rtpdefangles:
                    f.write('%9s %5s %5s' %angle + '\n')
            if self.rtpdefdihedrals:
                f.write('\n')
                f.write(' [ dihedrals ] \n')
                for dihedral in self.rtpdefdihedrals:
                    f.write('%9s %5s %5s %5s' %dihedral + '\n') 
            if self.rtpdefimps:
                f.write('\n')
                f.write(' [ impropers ] \n')
                for improper in self.rtpdefimps:
                    f.write('%9s %5s %5s %5s' %improper + '\n')
      
        
    def Draw(self, captions = '', ShowImpropers = '',              ShowMissingBonds = False, ShowMissingAngles = False, ShowMissingDihedrals = False):
        """Draws a residue with Matplotlib.        Possible captions: 'names', 'numbers', 'types', 'elements' """
        if ShowMissingBonds == True:
            missinglist = self.CheckBonds()
        plt.rcParams["figure.figsize"] = (20,15)
        colors = {'C': 'black', 'H': 'gray', 'N': 'b', 'O': 'r', 'P':'y', 'S': 'orange'}
        m = self.mol
        atnumtype = self.numtypes
        nametype = self.atomtypes
        namenums = self.atomnums
        Chem.SanitizeMol(m)
        AllChem.Compute2DCoords(m)
        order = 0
        fig, ax = plt.subplots()
        
        rtpimproperslist = []
        linearimproperslist = []
        roundimproperslist = []
        if ShowImpropers == 'rtp':
            rtpimproperslist = self.rtpdefimps
        elif ShowImpropers == 'itp':
            self. FindItpImpropers()
            linearimproperslist = self.itpfoundlinearimps
            roundimproperslist = self.itpfoundroundimps
        if rtpimproperslist:   # BAD REPRESENTATION, BAD
            for impnames in rtpimproperslist:
                pos = m.GetConformer().GetAtomPosition(namenums[impnames[0]])
                tmp = patches.Circle((pos.x, pos.y), radius = 0.7, color = 'b',  alpha = 0.3)
                ax.add_artist(tmp)
        if linearimproperslist:  # This one's okay tho
            for impnames in linearimproperslist:
                xi = []
                yi = []
                for impname in impnames:
                    posi = m.GetConformer().GetAtomPosition(namenums[impname])
                    xi.append(posi.x)
                    yi.append(posi.y)
                bezier(xi, yi, ax, 'g')
        if roundimproperslist:
            for impnames in roundimproperslist:
                pos = m.GetConformer().GetAtomPosition(namenums[impnames[0]])
                tmp = patches.Circle((pos.x, pos.y), radius = 0.7, color = 'g',  alpha = 0.3)
                ax.add_artist(tmp)
        
        if ShowMissingAngles == True:
            missingangles = self.CheckAngles()
            t = 1.7
            f = 0
            for angle in self.FindMolAngles():
                if f == 0:
                    t +=0.46
                    if t>2.9:
                        f = 1
                else:
                    t+=(-0.43)
                    if t<1.8:
                        f = 0
                a1type = nametype[angle[0]]
                a2type = nametype[angle[1]]
                a3type = nametype[angle[2]]
                if (a1type, a2type, a3type) in missingangles or (a3type, a2type, a1type) in missingangles:
                    pos1 = m.GetConformer().GetAtomPosition(namenums[angle[0]])
                    pos2 = m.GetConformer().GetAtomPosition(namenums[angle[1]])
                    pos3 = m.GetConformer().GetAtomPosition(namenums[angle[2]])
                    x = [pos1.x, pos2.x, pos3.x]
                    y = [pos1.y, pos2.y, pos3.y]
                    atomanglewedge(x, y, ax, t)
            
        if ShowMissingDihedrals == True:
            missingdihedrals = self.CheckDihedrals()  
            for dihedral in self.FindMolDihedrals():
                xi = []
                yi = []
                a1type = nametype[dihedral[0]]
                a2type = nametype[dihedral[1]]
                a3type = nametype[dihedral[2]]
                a4type = nametype[dihedral[3]]
                if  (a1type, a2type, a3type, a4type) in missingdihedrals or (a4type, a3type, a2type, a1type) in missingdihedrals:
                    for atname in dihedral:
                        posi = m.GetConformer().GetAtomPosition(namenums[atname])
                        xi.append(posi.x)
                        yi.append(posi.y)
                    bezier(xi, yi, ax, '#B22222')
            
        for bond in m.GetBonds():   
            at1 = bond.GetBeginAtomIdx()
            at2 = bond.GetEndAtomIdx()
            pos1 = m.GetConformer().GetAtomPosition(at1)
            pos2 = m.GetConformer().GetAtomPosition(at2)
            x1 = pos1.x
            y1 = pos1.y
            x2 = pos2.x
            y2 = pos2.y
            bcolor = 'black'
            if ShowMissingBonds == True:
                atype1 = atnumtype[at1]
                atype2 = atnumtype[at2]
                if (atype1, atype2) in missinglist:
                    bcolor = 'r'
            plt.plot((x1, x2), (y1, y2),  color=bcolor,zorder=order)
            order+=1
            
        x = []
        y = []
        color = []
        for atom in m.GetAtoms():  
            n = 0
            anum = atom.GetIdx()
            atype = GetAtomType(atnumtype[anum])
            aname = GetAtomName(atom)
            pos = m.GetConformer().GetAtomPosition(anum)
            text = ''
            x.append(pos.x)
            y.append(pos.y)
            color.append(colors[atype])
            if captions:
                captiondict = {'numbers': anum, 'names': aname, 'types': atnumtype[anum], 'elements': atype}
                ax.annotate(captiondict[captions], xy = (pos.x, pos.y), xytext=(3, 3), textcoords = 'offset points',                        bbox = dict(color = 'w', alpha = 0.3), zorder = order+2)
        plt.scatter(x, y, s = 200, c = color, zorder = order, edgecolors = 'black')
        plt.axis('scaled')
        plt.show()   

