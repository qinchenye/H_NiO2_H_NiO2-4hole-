import subprocess
import os
import sys
import time
import shutil
import math
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import time

import parameters as pam
import hamiltonian as ham
import lattice as lat
import variational_space as vs 
import utility as util


def reorder_z(slabel):
    '''
    reorder orbs such that d orb is always before p orb and Ni layer (z=1) before Cu layer (z=0)
    '''
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    
    state_label = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    
    if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs:
        if x2<x1:
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        elif x2==x1 and orb1=='dx2y2' and orb2=='d3z2r2':
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]          
           
    elif orb1 in pam.O_orbs and orb2 in pam.Ni_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        
    elif orb1 in pam.O_orbs and orb2 in pam.O_orbs:
        if z2>z1:
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        if z2==z1:
            if x2<x1:
                state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
            
    elif orb1 in pam.H_orbs and orb2 in pam.Ni_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]  
        
    elif orb1 in pam.O_orbs and orb2 in pam.H_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]           
            
    return state_label
                
def make_z_canonical(slabel):
    
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
    s4 = slabel[15]; orb4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];    
    '''
    For three holes, the original candidate state is c_1*c_2*c_3|vac>
    To generate the canonical_state:
    1. reorder c_1*c_2 if needed to have a tmp12;
    2. reorder tmp12's 2nd hole part and c_3 to have a tmp23;
    3. reorder tmp12's 1st hole part and tmp23's 1st hole part
    '''
    tlabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    tmp12 = reorder_z(tlabel)

    tlabel = tmp12[5:10]+[s3,orb3,x3,y3,z3]
    tmp23 = reorder_z(tlabel)

    tlabel = tmp12[0:5]+tmp23[0:5]
    tmp = reorder_z(tlabel)

    slabel = tmp+tmp23[5:10]
    tlabel = slabel[10:15] + [s4,orb4,x4,y4,z4]
    tmp34 = reorder_z(tlabel)
    
    if tmp34 == tlabel:
        slabel2 = slabel + [s4,orb4,x4,y4,z4]
    elif  tmp34 != tlabel:
        tlabel = slabel[5:10] + [s4,orb4,x4,y4,z4]
        tmp24 = reorder_z(tlabel)
        if tmp24 == tlabel:
            slabel2 = slabel[0:10]+ [s4,orb4,x4,y4,z4] + slabel[10:15]
        elif  tmp24 != tlabel:
            tlabel = slabel[0:5] + [s4,orb4,x4,y4,z4]   
            tmp14 = reorder_z(tlabel)
            if tmp14 == tlabel:
                slabel2 = slabel[0:5]+ [s4,orb4,x4,y4,z4] + slabel[5:15]
            elif  tmp14 != tlabel:
                slabel2 = [s4,orb4,x4,y4,z4] + slabel[0:15]     
                
                
    return slabel2




def get_ground_state(matrix, VS, S_000_val,Sz_000_val,S_200_val,Sz_200_val):  
    '''
    Obtain the ground state info, namely the lowest peak in Aw_dd's component
    in particular how much weight of various d8 channels: a1^2, b1^2, b2^2, e^2
    '''        
    t1 = time.time()
    print ('start getting ground state')
#     # in case eigsh does not work but matrix is actually small, e.g. Mc=1 (CuO4)
#     M_dense = matrix.todense()
#     #print 'H='
#     #print M_dense
    
#     for ii in range(0,1325):
#         for jj in range(0,1325):
#             if M_dense[ii,jj]>0 and ii!=jj:
#                 print ii,jj,M_dense[ii,jj]
#             if M_dense[ii,jj]==0 and ii==jj:
#                 print ii,jj,M_dense[ii,jj]
                    
                
#     vals, vecs = np.linalg.eigh(M_dense)
#     vals.sort()
#     print 'lowest eigenvalue of H from np.linalg.eigh = '
#     print vals
    
    # in case eigsh works:
    Neval = pam.Neval
    vals, vecs = sps.linalg.eigsh(matrix, k=Neval, which='SA')
    vals.sort()
    print ('lowest eigenvalue of H from np.linalg.eigsh = ')
    print (vals)
    
    print("---get_ground_state_eigsh %s seconds ---" % (time.time() - t1))
    
    # get state components in GS and another 9 higher states; note that indices is a tuple
    for k in range(0,1):
        #if vals[k]<pam.w_start or vals[k]>pam.w_stop:
        #if vals[k]<11.5 or vals[k]>14.5:
        #if k<Neval:
        #    continue
            
        print ('E',k,' = ', vals[k])
        indices = np.nonzero(abs(vecs[:,k])>0.05)
        
        # stores all weights for sorting later
        dim = len(vecs[:,k])
        allwgts = np.zeros(dim)
        allwgts = abs(vecs[:,k])**2
        ilead = np.argsort(-allwgts)   # argsort returns small value first by default
            
        wgt_d8d8 = np.zeros(20)
        wgt_d9Ld8 = np.zeros(20)  
        wgt_d8d9L = np.zeros(20)          
#         wgt_d9Ld9L = np.zeros(20) 
#         wgt_d9L2d9 = np.zeros(20)         
        wgt_d9d9L2 = np.zeros(20)           
        wgt_d9d10L3 = np.zeros(20)   
#         wgt_d10L2d9L = np.zeros(20)             
        wgt_ddHH = np.zeros(20)             
        wgt_dddH = np.zeros(20)            
        wgt_ddHL = np.zeros(20)   
        wgt_d = np.zeros(20)          
        
        total = 0
        total2 = 0
        print ("Compute the weights in GS (lowest Aw peak)")
        
        #for i in indices[0]:
        for i in range(dim):
            # state is original state but its orbital info remains after basis change
            istate = ilead[i]
            weight = allwgts[istate]
            
            #if weight>0.01:
            if total<0.999:
                total += weight
                
                state = VS.get_state(VS.lookup_tbl[istate])

                s1 = state['hole1_spin']
                s2 = state['hole2_spin']
                s3 = state['hole3_spin']
                s4 = state['hole4_spin']            
                orb1 = state['hole1_orb']
                orb2 = state['hole2_orb']
                orb3 = state['hole3_orb']
                orb4 = state['hole4_orb']            
                x1, y1, z1 = state['hole1_coord']
                x2, y2, z2 = state['hole2_coord']
                x3, y3, z3 = state['hole3_coord']
                x4, y4, z4 = state['hole4_coord']    

                # also obtain the total S and Sz of the state
                S_000_12  = S_000_val[istate]
                Sz_000_12 = Sz_000_val[istate]
                S_200_12  = S_200_val[istate]
                Sz_200_12 = Sz_200_val[istate]                
                
                
                slabel=[s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3,s4,orb4,x4,y4,z4]
                slabel= make_z_canonical(slabel)
                s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
                s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
                s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];
                s4 = slabel[15]; orb4 = slabel[16]; x4 = slabel[17]; y4 = slabel[18]; z4 = slabel[19];     
                       
                
                
                #if i in indices[0]:
                #if not(abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1. or abs(x3)>1. or abs(y3)>1.):
                if weight>0.0001:
                    print ('state ', istate, s1,orb1,x1,y1,z1,',',s2,orb2,x2,y2,z2,',',s3,orb3,x3,y3,z3,',',s4,orb4,x4,y4,z4, \
                       ', S_000=',S_000_12,'Sz_000=',Sz_000_12,  ', S_200=',S_200_12,'Sz_200=',Sz_200_12,", weight = ", weight)
                    total2 += weight
                
                # analyze the states on Ni at (0,0,0)
                

                if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs and orb3 in pam.Ni_orbs and x1==x2==0 and x3==x4==2:
                    wgt_d8d8[0] += weight
                    if orb1=='dx2y2' and  orb2=='dx2y2'  and  orb3=='dx2y2' and  orb4=='dx2y2':
                        wgt_d8d8[1] += weight
                        if S12==0:
                            wgt_d8d8[4] += weight
                    if orb1=='d3z2r2' and  orb2=='dx2y2'  and  orb3=='dx2y2' and orb4=='dx2y2':
                        wgt_d8d8[2] += weight          
                        if S12==1:
                            wgt_d8d8[5] += weight                        
                    if orb1=='d3z2r2' and  orb2=='dx2y2'  and  orb3=='d3z2r2'and orb4=='dx2y2':
                        wgt_d8d8[3] += weight
                        if S12==1:
                            wgt_d8d8[6] += weight                           
                        
                if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs and orb3 in pam.Ni_orbs and orb4 in pam.O_orbs and x1==0 and x2==x3==2 and ((x4==1 and y4==0) or (x4==-1 and y4==0) or (x4==0 and y4==-1) or (x4==0 and y4==1)):
                    wgt_d9Ld8[0] += weight
                    if orb1=='dx2y2' and  orb2=='dx2y2'  and  orb3=='dx2y2':
                        wgt_d9Ld8[1] += weight
                        if S12==0:
                            wgt_d9Ld8[4] += weight 
                    if orb1=='d3z2r2' and  orb2=='dx2y2'  and  orb3=='dx2y2':
                        wgt_d9Ld8[2] += weight          
                        if S12==1:
                            wgt_d9Ld8[5] += weight                             
                    if orb1=='d3z2r2' and  orb2=='d3z2r2'  and  orb3=='dx2y2':
                        wgt_d9Ld8[3] += weight          
                        if S12==0:
                            wgt_d9Ld8[6] += weight      
                            
                if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs and orb3 in pam.Ni_orbs and orb4 in pam.O_orbs and x1==x2==0 and x3==2 and ((x4==1 and y4==0) or (x4==-1 and y4==0) or (x4==0 and y4==-1) or (x4==0 and y4==1)):
                    wgt_d8d9L[0] += weight                            
#                     if orb1=='dx2y2' and  orb2=='dx2y2':
#                         wgt_d9Ld9[1] += weight
#                     if orb1=='d3z2r2' and  orb2=='dx2y2':
#                         wgt_d9Ld9[2] += weight
#                     if orb1=='dx2y2' and  orb2=='d3z2r2':
#                         wgt_d9Ld9[3] += weight
#                     if orb1=='d3z2r2' and  orb2=='d3z2r2':
#                         wgt_d9Ld9[4] += weight      
     
                if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs and orb3 in pam.O_orbs and orb4 in pam.O_orbs and x1==0 and x2==2:
                    wgt_d9d9L2[0] += weight   
                    if (abs(x3-0) + abs(y3-0))==1 and (abs(x4-0) + abs(y4-0))==1:
                        wgt_d9d9L2[1] += weight
                        if orb1=='d3z2r2' and orb2=='d3z2r2':
                            wgt_d9d9L2[4] += weight
                        if orb1=='d3z2r2' and orb2=='dx2y2':
                            wgt_d9d9L2[5] += weight                            
                        if orb1=='dx2y2' and orb2=='d3z2r2':
                            wgt_d9d9L2[6] += weight                                
                        if orb1=='dx2y2' and orb2=='dx2y2':
                            wgt_d9d9L2[7] += weight      
                    elif (abs(x3-0) + abs(y3-0))==1 and ((abs(x4-2) + abs(y4-0))==1 and (x4,y4)!=(2,0)):
                        wgt_d9d9L2[2] += weight  
                        if orb1=='d3z2r2' and orb2=='d3z2r2':
                            wgt_d9d9L2[8] += weight
                        if orb1=='d3z2r2' and orb2=='dx2y2':
                            wgt_d9d9L2[9] += weight                            
                        if orb1=='dx2y2' and orb2=='d3z2r2':
                            wgt_d9d9L2[10] += weight                                
                        if orb1=='dx2y2' and orb2=='dx2y2':
                            wgt_d9d9L2[11] += weight                             
                    elif ((abs(x3-2) + abs(y3-0))==1 and (x3,y3)!=(2,0)) and ((abs(x4-2) + abs(y4-0))==1 and (x4,y4)!=(2,0)):
                        wgt_d9d9L2[3] += weight                          
                        
                        
#                     if orb1=='dx2y2' and  orb2=='dx2y2':
#                         wgt_d9d9L[1] += weight
#                     if orb1=='d3z2r2' and  orb2=='dx2y2':
#                         wgt_d9d9L[2] += weight
#                     if orb1=='dx2y2' and  orb2=='d3z2r2':
#                         wgt_d9d9L[3] += weight
#                     if orb1=='d3z2r2' and  orb2=='d3z2r2':
#                         wgt_d9d9L[4] += weight   
 
                if orb1 in pam.Ni_orbs and orb2 in pam.O_orbs and orb3 in pam.O_orbs and orb4 in pam.O_orbs:
                    wgt_d9d10L3[0] += weight  
                    if x1==0:
                        if (abs(x2-0) + abs(y2-0))==1 and (abs(x3-0) + abs(y3-0))==1 and (abs(x4-0) + abs(y4-0))==1:
                            wgt_d9d10L3[1] += weight                 
                        if (abs(x2-0) + abs(y2-0))==1 and (abs(x3-0) + abs(y3-0))==1 and ((abs(x4-2) + abs(y4-0))==1 and (x4,y4)!=(2,0)):
                            wgt_d9d10L3[2] += weight                    
                        if (abs(x2-0) + abs(y2-0))==1 and ((abs(x3-2) + abs(y3-0))==1 and (x3,y3)!=(2,0)) and ((abs(x4-2) + abs(y4-0))==1 and (x4,y4)!=(2,0)):
                            wgt_d9d10L3[3] += weight    
                        if ((abs(x2-2) + abs(y2-0))==1 and (x2,y2)!=(2,0)) and ((abs(x3-2) + abs(y3-0))==1 and (x3,y3)!=(2,0)) and ((abs(x4-2) + abs(y4-0))==1 and (x4,y4)!=(2,0)):
                            wgt_d9d10L3[4] += weight                            
                    if x1==2:
                        if (abs(x2-0) + abs(y2-0))==1 and (abs(x3-0) + abs(y3-0))==1 and (abs(x4-0) + abs(y4-0))==1:
                            wgt_d9d10L3[5] += weight                 
                        if (abs(x2-0) + abs(y2-0))==1 and (abs(x3-0) + abs(y3-0))==1 and ((abs(x4-2) + abs(y4-0))==1 and (x4,y4)!=(2,0)):
                            wgt_d9d10L3[6] += weight                    
                        if (abs(x2-0) + abs(y2-0))==1 and ((abs(x3-2) + abs(y3-0))==1 and (x3,y3)!=(2,0)) and ((abs(x4-2) + abs(y4-0))==1 and (x4,y4)!=(2,0)):
                            wgt_d9d10L3[7] += weight    
                        if ((abs(x2-2) + abs(y2-0))==1 and (x2,y2)!=(2,0)) and ((abs(x3-2) + abs(y3-0))==1 and (x3,y3)!=(2,0)) and ((abs(x4-2) + abs(y4-0))==1 and (x4,y4)!=(2,0)):
                            wgt_d9d10L3[8] += weight                               
                        
#                     if orb1=='dx2y2' and  orb2=='dx2y2':
#                         wgt_d9d9H[1] += weight
#                     if orb1=='d3z2r2' and  orb2=='dx2y2':
#                         wgt_d9d9H[2] += weight
#                     if orb1=='dx2y2' and  orb2=='d3z2r2':
#                         wgt_d9d9H[3] += weight
#                     if orb1=='d3z2r2' and  orb2=='d3z2r2':
#                         wgt_d9d9H[4] += weight   





                                                                                                 
                if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs and orb3 in pam.H_orbs and orb4 in pam.H_orbs:
                    wgt_ddHH[0] += weight           
                    
                if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs and orb3 in pam.Ni_orbs and orb4 in pam.H_orbs:
                    wgt_dddH[0] += weight                       
                    
                if orb1 in pam.Ni_orbs and orb2 in pam.Ni_orbs and orb3 in pam.H_orbs and orb4 in pam.O_orbs:
                    wgt_ddHL[0] += weight
                    if x1==0 and x2==0:
                        wgt_ddHL[1] += weight
                    elif x1==0 and x2==2:
                        wgt_ddHL[2] += weight
                        if orb1=='d3z2r2' and orb2=='d3z2r2':
                            wgt_ddHL[4] += weight
                        if orb1=='d3z2r2' and orb2=='dx2y2':
                            wgt_ddHL[5] += weight                            
                        if orb1=='dx2y2' and orb2=='d3z2r2':
                            wgt_ddHL[6] += weight                                
                        if orb1=='dx2y2' and orb2=='dx2y2':
                            wgt_ddHL[7] += weight                                
                    elif x1==2 and x2==2:
                        wgt_ddHL[3] += weight                         
                        
                    
                if orb1 in pam.Ni_orbs and orb2 in pam.H_orbs :
                    wgt_d[0] += weight                        
                    if orb3 in pam.H_orbs:
                        wgt_d[1] += weight
                        if x1==0:
                            wgt_d[9] += weight
                            if orb1=='d3z2r2':
                                wgt_d[11] += weight
                            if orb1=='dx2y2':
                                wgt_d[12] += weight                                   
                        if x1==2:
                            wgt_d[10] += weight                        
                            if orb1=='d3z2r2':
                                wgt_d[13] += weight
                            if orb1=='dx2y2':
                                wgt_d[14] += weight                            
                        
                    if orb3 in pam.O_orbs:
                        wgt_d[2] += weight
                        if x1==0:
                            wgt_d[3] += weight
                            if orb1=='d3z2r2':
                                wgt_d[5] += weight
                            if orb1=='dx2y2':
                                wgt_d[6] += weight                                   
                        if x1==2:
                            wgt_d[4] += weight                        
                            if orb1=='d3z2r2':
                                wgt_d[7] += weight
                            if orb1=='dx2y2':
                                wgt_d[8] += weight                             
                        
        print('printed states total weight =', total)
        
        print('wgt_d8d8 = ',wgt_d8d8[0])
        print('wgt_d9Ld8= ',wgt_d9Ld8[0])
        print('wgt_d8d9L = ',wgt_d8d9L[0])
        print('wgt_d9d9L2 = ',wgt_d9d9L2[3])
        print('s = ',wgt_d9d9L2[0])        
        print('wgt_d9L2d9 = ',wgt_d9d9L2[1])        
        print('wgt_d9Ld9L = ',wgt_d9d9L2[2]) 
        print('S2 = ',wgt_d9d10L3)          
        print('wgt_d9L3d10 = ',wgt_d9d10L3[1])           
        print('wgt_d9L2d10L = ',wgt_d9d10L3[2])         
        print('wgt_d9Ld10L2 = ',wgt_d9d10L3[3])        
        print('wgt_d9d10L3 = ',wgt_d9d10L3[4])
        
        
        print('wgt_ddHH = ',wgt_ddHH[0])
        print('wgt_dddH = ',wgt_dddH[0])
        print('wgt_ddHL = ',wgt_ddHL[0])        
        print('wgt_d = ',wgt_d[0])        
        
        print('total weight = ', wgt_d8d8[0]+ wgt_d9Ld8[0]+wgt_d8d9L[0]+ wgt_d9d9L2[0]+wgt_d9d10L3[0]+wgt_ddHH[0]\
                          +wgt_dddH[0] +wgt_ddHL[0] +wgt_d[0])
        print('total2 = ',total2)  
        
        
        
        path = './data'		# create file

        if os.path.isdir(path) == False:
            os.mkdir(path) 
        txt=open('./data/wgt_d8d8','a')                                  
        txt.write(str(wgt_d8d8[0])+'\n')
        txt.close() 
        txt=open('./data/wgt_d8d8_b1b1_b1b1','a')                                  
        txt.write(str(wgt_d8d8[1])+'\n')
        txt.close() 
        txt=open('./data/wgt_d8d8_a1b1_b1b1','a')                                  
        txt.write(str(wgt_d8d8[2])+'\n')
        txt.close()         
        txt=open('./data/wgt_d8d8_a1b1_a1b1','a')                                  
        txt.write(str(wgt_d8d8[3])+'\n')
        txt.close()              
        txt=open('./data/wgt_d8d8_b1b1_b1b1_0','a')                                  
        txt.write(str(wgt_d8d8[4])+'\n')
        txt.close()  
        txt=open('./data/wgt_d8d8_a1b1_b1b1_1','a')                                  
        txt.write(str(wgt_d8d8[5])+'\n')
        txt.close()         
        txt=open('./data/wgt_d8d8_a1b1_a1b1_1','a')                                  
        txt.write(str(wgt_d8d8[6])+'\n')
        txt.close()          
        
        
        txt=open('./data/wgt_d9Ld8','a')                                  
        txt.write(str(wgt_d9Ld8[0])+'\n')
        txt.close()
        txt=open('./data/wgt_d9Ld8_b1_b1b1','a')                                  
        txt.write(str(wgt_d9Ld8[1])+'\n')
        txt.close()          
        txt=open('./data/wgt_d9Ld8_a1_b1b1','a')                                  
        txt.write(str(wgt_d9Ld8[2])+'\n')
        txt.close()   
        txt=open('./data/wgt_d9Ld8_a1_a1b1','a')                                  
        txt.write(str(wgt_d9Ld8[3])+'\n')
        txt.close()          
        txt=open('./data/wgt_d9Ld8_b1_b1b1_0','a')                                  
        txt.write(str(wgt_d9Ld8[4])+'\n')
        txt.close()          
        txt=open('./data/wgt_d9Ld8_a1_b1b1_1','a')                                  
        txt.write(str(wgt_d9Ld8[5])+'\n')
        txt.close()   
        txt=open('./data/wgt_d9Ld8_a1_a1b1_0','a')                                  
        txt.write(str(wgt_d9Ld8[6])+'\n')
        txt.close()           
        

        txt=open('./data/wgt_d8d9L','a')                                  
        txt.write(str(wgt_d8d9L[0])+'\n')
        txt.close()        
        
        txt=open('./data/wgt_d9d9L2_sum','a')                                  
        txt.write(str(wgt_d9d9L2[0])+'\n')
        txt.close()          
        txt=open('./data/wgt_d9L2d9','a')                                  
        txt.write(str(wgt_d9d9L2[1])+'\n')
        txt.close()          
        txt=open('./data/wgt_d9Ld9L','a')                                  
        txt.write(str(wgt_d9d9L2[2])+'\n')
        txt.close()          
        txt=open('./data/wgt_d9d9L2','a')                                  
        txt.write(str(wgt_d9d9L2[3])+'\n')
        txt.close()           
        txt=open('./data/wgt_d9L2d9_a1a1','a')                                  
        txt.write(str(wgt_d9d9L2[4])+'\n')
        txt.close()         
        txt=open('./data/wgt_d9L2d9_a1b1','a')                                  
        txt.write(str(wgt_d9d9L2[5])+'\n')
        txt.close()                 
        txt=open('./data/wgt_d9L2d9_b1a1','a')                                  
        txt.write(str(wgt_d9d9L2[6])+'\n')
        txt.close()                 
        txt=open('./data/wgt_d9L2d9_b1b1','a')                                  
        txt.write(str(wgt_d9d9L2[7])+'\n')
        txt.close()             
        txt=open('./data/wgt_d9Ld9L_a1a1','a')                                  
        txt.write(str(wgt_d9d9L2[8])+'\n')
        txt.close()         
        txt=open('./data/wgt_d9Ld9L_a1b1','a')                                  
        txt.write(str(wgt_d9d9L2[9])+'\n')
        txt.close()                 
        txt=open('./data/wgt_d9Ld9L_b1a1','a')                                  
        txt.write(str(wgt_d9d9L2[10])+'\n')
        txt.close()                 
        txt=open('./data/wgt_d9Ld9L_b1b1','a')                                  
        txt.write(str(wgt_d9d9L2[11])+'\n')
        txt.close()                 
        
        
        
        
        txt=open('./data/wgt_d9d10L3_sum','a')                                  
        txt.write(str(wgt_d9d10L3[0])+'\n')
        txt.close()    
        txt=open('./data/wgt_d9L3d10','a')                                  
        txt.write(str(wgt_d9d10L3[1])+'\n')
        txt.close()            
        txt=open('./data/wgt_d9L2d10L','a')                                  
        txt.write(str(wgt_d9d10L3[2])+'\n')
        txt.close()              
        txt=open('./data/wgt_d9Ld10L2','a')                                  
        txt.write(str(wgt_d9d10L3[3])+'\n')
        txt.close()              
        txt=open('./data/wgt_d9d10L3','a')                                  
        txt.write(str(wgt_d9d10L3[4])+'\n')
        txt.close()     
        txt=open('./data/wgt_d10L3d9','a')                                  
        txt.write(str(wgt_d9d10L3[5])+'\n')
        txt.close()            
        txt=open('./data/wgt_d10L2d9L','a')                                  
        txt.write(str(wgt_d9d10L3[6])+'\n')
        txt.close()              
        txt=open('./data/wgt_d10Ld9L2','a')                                  
        txt.write(str(wgt_d9d10L3[7])+'\n')
        txt.close()              
        txt=open('./data/wgt_d10d9L3','a')                                  
        txt.write(str(wgt_d9d10L3[8])+'\n')
        txt.close()            
        
        txt=open('./data/wgt_ddHL_sum','a')                                  
        txt.write(str(wgt_ddHL[0])+'\n')
        txt.close()   
        txt=open('./data/wgt_ddHL_d8d10HL','a')                                  
        txt.write(str(wgt_ddHL[1])+'\n')
        txt.close()   
        txt=open('./data/wgt_ddHL_d9d9HL','a')                                  
        txt.write(str(wgt_ddHL[2])+'\n')
        txt.close()   
        txt=open('./data/wgt_ddHL_d10d8HL','a')                                  
        txt.write(str(wgt_ddHL[3])+'\n')
        txt.close()
        txt=open('./data/wgt_ddHL_d9d9HL_a1a1','a')                                  
        txt.write(str(wgt_ddHL[4])+'\n')
        txt.close()          
        txt=open('./data/wgt_ddHL_d9d9HL_a1b1','a')                                  
        txt.write(str(wgt_ddHL[5])+'\n')
        txt.close()                 
        txt=open('./data/wgt_ddHL_d9d9HL_b1a1','a')                                  
        txt.write(str(wgt_ddHL[6])+'\n')
        txt.close()                 
        txt=open('./data/wgt_ddHL_d9d9HL_b1b1','a')                                  
        txt.write(str(wgt_ddHL[7])+'\n')
        txt.close()                 
        
        
        
        txt=open('./data/wgt_d_sum','a')                                  
        txt.write(str(wgt_d[0])+'\n')
        txt.close()        
        txt=open('./data/wgt_dHHL','a')                                  
        txt.write(str(wgt_d[1])+'\n')
        txt.close()        
        txt=open('./data/wgt_dHLL','a')                                  
        txt.write(str(wgt_d[2])+'\n')
        txt.close()        
        txt=open('./data/wgt_dHLL_d9d10','a')                                  
        txt.write(str(wgt_d[3])+'\n')
        txt.close() 
        txt=open('./data/wgt_dHLL_d10d9','a')                                  
        txt.write(str(wgt_d[4])+'\n')
        txt.close()         
        txt=open('./data/wgt_dHLL_d9d10_a1','a')                                  
        txt.write(str(wgt_d[5])+'\n')
        txt.close()         
        txt=open('./data/wgt_dHLL_d9d10_b1','a')                                  
        txt.write(str(wgt_d[6])+'\n')
        txt.close()         
        txt=open('./data/wgt_dHLL_d10d9_a1','a')                                  
        txt.write(str(wgt_d[7])+'\n')
        txt.close()         
        txt=open('./data/wgt_dHLL_d10d9_b1','a')                                  
        txt.write(str(wgt_d[8])+'\n')
        txt.close()                 
        txt=open('./data/wgt_dHHL_d9d10','a')                                  
        txt.write(str(wgt_d[9])+'\n')
        txt.close() 
        txt=open('./data/wgt_dHHL_d10d9','a')                                  
        txt.write(str(wgt_d[10])+'\n')
        txt.close()         
        txt=open('./data/wgt_dHHL_d9d10_a1','a')                                  
        txt.write(str(wgt_d[11])+'\n')
        txt.close()         
        txt=open('./data/wgt_dHHL_d9d10_b1','a')                                  
        txt.write(str(wgt_d[12])+'\n')
        txt.close()         
        txt=open('./data/wgt_dHHL_d10d9_a1','a')                                  
        txt.write(str(wgt_d[13])+'\n')
        txt.close()         
        txt=open('./data/wgt_dHHL_d10d9_b1','a')                                  
        txt.write(str(wgt_d[14])+'\n')
        txt.close()         
        
        
        
        
    return vals #, vecs, wgt_d8, wgt_d9L, wgt_d10L2