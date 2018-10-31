# GIFTEd
Gromacs Interactive Force-field & Topology Editor
### Annotation
This library allows to:
- read .rtp and .itp files to create Residue and Force Field objects;
- modify Residue molecule by adding and removing atoms;
- visualise Residue molecule and show its non-parametrized bonds, angles, dihedrals & impropers;
- add parameters to FF;
- write new .rtp and .itp files.



### What else do you need?
In order for GIFtEd to work, you should have following packages installed:
- RDKit;
- NumPy;
- Matplotlib.


### Current issues:
- pdb writer + pdb2gmx check
- comments, empty slots, base class of ff parameters (func type and comments included)
- charges 
- writing an rtp header with basic func types 
- pairs for VdW for non-bonded
- all other kinds of files?? (hdb and stuff)
