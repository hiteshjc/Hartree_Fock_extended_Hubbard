import os
import copy
import sys
import numpy as N
import pylab


#############################################################################
#Code by Hitesh J C, July 14, 2026
#############################################################################
def make_bravais_lattice(L1,L2,vec1,vec2):
    coords=[]
    nearest_neighbor_pairs=[]
    for n2 in range(L2):
        for n1 in range(L1):
            x=(n1*vec1[0])+(n2*vec2[0])
            y=(n1*vec1[1])+(n2*vec2[1])
            coords.append([n1,n2,x,y])
    for i in range(0,len(coords)):
        for j in range(i+1,len(coords)):
            n1i=coords[i][0]
            n2i=coords[i][1]
            xi=coords[i][2]
            yi=coords[i][3]

            n1j=coords[j][0]
            n2j=coords[j][1]
            xj=coords[j][2]
            yj=coords[j][3]
            dist1=N.sqrt(((xi-xj)*(xi-xj)) + ((yi-yj)*(yi-yj)))
            
            xjnew=((n1j-L1)*vec1[0])+(n2j*vec2[0])
            yjnew=((n1j-L1)*vec1[1])+(n2j*vec2[1])
            dist2=N.sqrt(((xi-xjnew)*(xi-xjnew)) + ((yi-yjnew)*(yi-yjnew)))
            
            xjnew=(n1j*vec1[0])+(n2j-L2)*vec2[0]
            yjnew=(n1j*vec1[1])+(n2j-L2)*vec2[1]
            dist3=N.sqrt(((xi-xjnew)*(xi-xjnew)) + ((yi-yjnew)*(yi-yjnew)))
            
            xjnew=((n1j+L1)*vec1[0])+(n2j*vec2[0])
            yjnew=((n1j+L1)*vec1[1])+(n2j*vec2[1])
            dist4=N.sqrt(((xi-xjnew)*(xi-xjnew)) + ((yi-yjnew)*(yi-yjnew)))
            
            xjnew=(n1j*vec1[0])+((n2j+L2)*vec2[0])
            yjnew=(n1j*vec1[1])+((n2j+L2)*vec2[1])
            dist5=N.sqrt(((xi-xjnew)*(xi-xjnew)) + ((yi-yjnew)*(yi-yjnew)))
            
            xjnew=((n1j+L1)*vec1[0])+((n2j-L2)*vec2[0])
            yjnew=((n1j+L1)*vec1[1])+((n2j-L2)*vec2[1])
            dist6=N.sqrt(((xi-xjnew)*(xi-xjnew)) + ((yi-yjnew)*(yi-yjnew)))
            
            dist=min(dist1,dist2,dist3,dist4,dist5,dist6)
            
            #if (abs(dist-1.0)<1.0e-10 and [i,j]!=[1,2]):nearest_neighbor_pairs.append([i,j]) 
            if (abs(dist-1.0)<1.0e-10):nearest_neighbor_pairs.append([i,j]) 
            #if (abs(dist-1.0)<1.0e-10 or abs(dist-(L1-1))<1.0e-10 or abs(dist-(L2-1))<1.0e-10):nearest_neighbor_pairs.append([i,j])  # Open BC
    return coords,nearest_neighbor_pairs
#############################################################################


def setup_and_solve_hf(nsites,nparticles,pairs,t,V,old_one_body):
    
    hf=N.zeros((nsites,nsites),dtype=complex)
    for pair in pairs:
        site1=pair[0]
        site2=pair[1]
        hf[site1,site1]+=(V*old_one_body[site2,site2]) # Hartree term
        hf[site2,site2]+=(V*old_one_body[site1,site1]) # Hartree term
        hf[site1,site2]= -t                            # Bare hopping
        hf[site2,site1]= -t                            # Bare hopping
        hf[site1,site2]+= (-V)*old_one_body[site2,site1] # Fock exchange 
        hf[site2,site1]+= (-V)*old_one_body[site1,site2] # Fock exchange
    
    eigs,vecs=N.linalg.eigh(hf)
    #print(eigs)

    new_one_body=N.zeros((nsites,nsites),dtype=complex)

    # Decide the fermi energy
    ef=eigs[nparticles-1]
    
    for site in range(nsites):
        for e in range(nsites):
            if (eigs[e]<=ef or abs(eigs[e]-ef)<1.0e-7): new_one_body[site,site]+=(abs(vecs[site,e])**2.0)
                
    for pair in pairs:
        site1=pair[0]
        site2=pair[1]
        for e in range(nsites):
            if (eigs[e]<=ef or abs(eigs[e]-ef)<1.0e-7):
                new_one_body[site1,site2]+=(N.conjugate(vecs[site1,e])*(vecs[site2,e]))
                new_one_body[site2,site1]+=(N.conjugate(vecs[site2,e])*(vecs[site1,e]))
    
    total_energy=0.0
    for e in range(len(eigs)):
            if (eigs[e]<=ef or abs(eigs[e]-ef)<1.0e-7): total_energy+=eigs[e]
   
    for pair in pairs: 
        i=pair[0]
        j=pair[1]
        total_energy+= (-V*new_one_body[i,i]*new_one_body[j,j]) # From Hartree term
        total_energy+= (V*new_one_body[i,j]*new_one_body[j,i])      # From Fock term

    return total_energy.real,new_one_body
#############################################################################

def get_error(nsites,new_one_body,old_one_body):
    err=0.0
    for i in range(nsites):err+=abs(new_one_body[i,i]-old_one_body[i,i])**2.0  # We are just computing error in densities
    return N.sqrt(err)

#############################################################################

vec1=[0,0]
vec2=[0,0]
lattice=sys.argv[1]
if (lattice=="square"):
    vec1=[1.0,0.0]
    vec2=[0.0,1.0]
if (lattice=="triangular"):
    vec1=[1.0,0.0]
    vec2=[1.0/2.0,N.sqrt(3.0)/2.0]
if (lattice=="oned"):
    vec1=[1.0,0.0]
    vec2=[0.0,0.0]

#############################
# PARAMETERS
#############################
filling=float(sys.argv[2])
t=1.0
V=float(sys.argv[3])
L1=int(sys.argv[4])
L2=int(sys.argv[5])
seed=int(sys.argv[6])
N.random.seed(seed)
nsites=L1*L2
nparticles=int(filling*nsites)
print("Nparticles = ",nparticles)
#initialize HF in real space
coords,pairs=make_bravais_lattice(L1,L2,vec1,vec2)
print(pairs)
nsites=len(coords)
nbonds=len(pairs)
print("Len coords = ",nsites)
print("Len pairs = ",nbonds)
#stop

################################
# INITIAL HOPPING and densities
################################
initial_hopping_matrix=N.zeros((nsites,nsites),dtype=complex)
for pair in pairs:
    # HOPS
    initial_hopping_matrix[pair[0],pair[1]]=-t
    initial_hopping_matrix[pair[1],pair[0]]=-t
elowest=100000
maxiter=3000

#################################################
# TRY HF multiple times and take the best result
#################################################
best_one_body=N.zeros((nsites,nsites),dtype=complex)
for ntries in range(1):
    print("Ntry = ",ntries)
    densities=N.zeros((nsites),dtype=float)
    # Initialize hopping and diagonals
    for i in range(nsites): densities[i]=filling+(0.2*(2*N.random.random()-1))
    #for i in range(nsites): densities[i]=filling
    total_number=sum(densities)
    old_one_body=copy.deepcopy(initial_hopping_matrix)  
    new_one_body=N.zeros((nsites,nsites),dtype=complex)
    rescale_factor=nparticles/total_number
    for i in range(nsites): old_one_body[i,i]=densities[i]*rescale_factor
    
    # Iterate - Use the old cidag cj to set up the new matrix 
    do_next=True
    for iteration in range(maxiter):
        if (do_next):
            total_energy,new_one_body=setup_and_solve_hf(nsites,nparticles,pairs,t,V,old_one_body)
            temp_one_body=N.zeros((nsites,nsites),dtype=complex)
            if(iteration>=2):
                for i in range(nsites):
                    for j in range(nsites):
                        temp_one_body[i,j]=(0.25*new_one_body[i,j])+(0.75*old_one_body[i,j])
            if (iteration<2): temp_one_body=copy.deepcopy(new_one_body)
            err = get_error(nsites,temp_one_body,old_one_body)
            if ((iteration%100==0 or iteration==1) and iteration!=0): print("Iter,Total_energy, Err = ",iteration,",",total_energy,",",err)
            if (iteration>1000 and err>1): do_next=False
            if (do_next): old_one_body=copy.deepcopy(temp_one_body)
    if ((total_energy<elowest and err<1.0e-2) or (iteration<=2)):
        best_one_body=copy.deepcopy(new_one_body)
        elowest=total_energy

###########################
# Print the densities
###########################
print("Lowest energy = ",elowest)
print("Bond")
for pair in pairs:   
    site1=pair[0]
    site2=pair[1]
    hartree=best_one_body[site1,site1]*best_one_body[site2,site2]
    fock=best_one_body[site1,site2]*best_one_body[site2,site1]
    print("%3d %3d %+5.15f %+5.15f" %(site1,site2,hartree,fock))
print("Site      x                       y                    <n>")
total_charge=0.0
for i in range(nsites):
    x=coords[i][2]
    y=coords[i][3]
    print("%3d  %+5.15f %+5.15f %+5.15f" %(i,x,y,best_one_body[i,i]))
    total_charge+=best_one_body[i,i]
    pylab.scatter(x,y,marker="o",s=best_one_body[i,i]*100,color="black")
print("Total charge =",total_charge)
pylab.show()
#print("Nu_tot=",round(sum(best_up_densities),2),"Nd_tot=",round(sum(best_down_densities),2))
#print("Sx2+Sy2+Sz2 = ",totsx*totsx + totsy*totsy + totsz*totsz)
