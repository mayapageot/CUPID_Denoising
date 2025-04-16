import numpy as np
import torch
def transform_pos(dir,filename):
    pos_injection = np.loadtxt(dir+filename+"_bis.pos")
    if pos_injection.shape[1]==3:
        np.savetxt(dir+filename+".trig",pos_injection[:,:2],fmt=' %16f, %.6f')
    elif pos_injection.shape[1]==7:
        pos_injection[:,1] +=  pos_injection[:,2]
        np.savetxt(dir+filename + ".trig", pos_injection[:,:2], fmt = ' %16f, %.6f')
    else:
        print(pos_injection.shape[1])
def from_pt_to_trig(dir,filename):
    print("input: ",dir+filename+".pos_selected.pt")
    pos_injection = torch.load(dir+filename+".pos_selected.pt").numpy().reshape(1,-1).T*4
    zeros_column = np.zeros((pos_injection.shape[0], 1))
    pos_injection_extended = np.hstack((pos_injection, zeros_column))
    print("output: ", dir + filename + "_selected.trig")
    np.savetxt(dir+filename+"_selected.trig",pos_injection_extended,fmt=' %16f, %.6f')


def from_pos_to_pt(dir,filename):
    a = torch.from_numpy((np.loadtxt(dir+filename+".pos")[:, 0] / 4).astype(int))
    torch.save(a, dir+filename+".pos.pt")
for i in [0,8]:
    for channel in [11]:
        partial_num = 1 if channel == 17 else 0
        dir = f"/feynman/work/projets/cupid/data/LSM/RUN9_pulse_injected/channel{channel}/"
        filename = f"pup_n1-d{i}_000813_20230628T161508_{channel:03}_{partial_num:03}"
        transform_pos(dir,filename)
        #from_pos_to_pt(dir,filename)
        # from_pt_to_trig(dir,filename)