# -*- coding: utf-8 -*-
"""
# ===== CIVL4145 Modelling Project:
# ===== Base model (preliminary model) - Pumping from a unconfined aquifer
# ===== with uniform background flow.
# =====

# updated from seawat to MF6 on 240810

"""

import flopy
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib as mpl


# Check versions
print('  ..... system version: ',sys.version)
print('  ..... numpy version: {}'.format(np.__version__))
print('  ..... matplotlib version: {}'.format(mpl.__version__))
print('  ..... flopy version: {}'.format(flopy.__version__))
print('  ..... pakages and modules imported.')


# Create the simulation object
sim = flopy.mf6.MFSimulation(sim_name='mf6_pumping_model', exe_name='mf6', version='mf6', sim_ws='.')

# Time discretization settings
tdis = flopy.mf6.ModflowTdis(sim, time_units='days', nper=2, perioddata=[(50*365, 50, 1.0), (100*365, 50, 1.0)])

# Create groundwater flow model
modelName = 'pumping_model'
model_nam_file = 'pumping.nam'
gwf = flopy.mf6.ModflowGwf(sim, modelname=model_nam_file, save_flows=True)

# Discretization package (DIS) - structured grid
nlay, nrow, ncol = 1, 100, 102  
delr = 100  # cell size along rows
delc = 100  # cell size along columns
top = 0
botm = [-80]  # bottom of layer
dis = flopy.mf6.ModflowGwfdis(gwf, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc, top=top, botm=botm)

# Initial conditions (IC)
strt = np.full((nlay, nrow, ncol), -35)  # initial head is -35 everywhere
strt[:, :, 0] = -30  # Left boundary initial head
strt[:, :, -1] = -40  # Right boundary initial head
ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)

# Constant Head (CHD) package
chd_spd = [[(0, i, 0), -30] for i in range(nrow)] + [[(0, i, ncol-1), -40] for i in range(nrow)]
chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data={0: chd_spd, 1: chd_spd})

# Node Property Flow (NPF) package for hydraulic conductivity
k = 3.0  # horizontal hydraulic conductivity
npf = flopy.mf6.ModflowGwfnpf(gwf, save_flows=True, icelltype=1, k=k)

# Storage (STO) package for unconfined aquifers
sto = flopy.mf6.ModflowGwfsto(gwf, 
                              iconvert=1, 
                              ss=1e-5, 
                              sy=0.10, 
                              steady_state={0: False}, 
                              transient={1: True})

# Well package (WEL)
wel_spd = {1: [[(0, 50, 50), -2000]]}  # Well pumping rate at center of the model

wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_spd)

# Output control (OC) package
oc = flopy.mf6.ModflowGwfoc(gwf, budget_filerecord='pumping_model.cbc', head_filerecord='pumping_model.hds',
                            saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')],
                            printrecord=[('HEAD', 'LAST'), ('BUDGET', 'LAST')])

# Solver
nouter, ninner = 100, 300
hclose, rclose, relax = 1e-6, 1e-6, 1.0

# ===== 2.3 Defining MODFLOW 6 solver for flow model
ims = flopy.mf6.ModflowIms(sim,
                              print_option="SUMMARY",
                              outer_dvclose=hclose,
                              outer_maximum=nouter,
                              under_relaxation="NONE",
                              inner_maximum=ninner,
                              inner_dvclose=hclose,
                              rcloserecord=rclose,
                              linear_acceleration="BICGSTAB",
                              scaling_method="NONE",
                              reordering_method="NONE",
                              relaxation_factor=relax,
                              filename="{}.ims".format(model_nam_file),)



#ims = flopy.mf6.ModflowIms(sim, print_option='ALL', outer_hclose=1e-5)

# Write and run the simulation
sim.write_simulation()
sim.run_simulation()




#%%==============================================================================
# 5.0 ===== POST-PROCESSING MODEL RESULTS =====================================
#==============================================================================
print(' ')
print('  Plotting results.....')
# 5.1 ===== Setting plot formatting (global settings - applies to all plots) ==
mS = 12 # Used to set marker size
lW = 3 # Used to set linewidth
fS = 18 # Used to set font size
plt.rcParams['font.family'] = 'Times New Roman' # Globally sets the font type
plt.rc('font',size=fS)
plt.rc('axes',titlesize=fS)
plt.rc('axes',labelsize=fS)
plt.rc('xtick',labelsize=fS)
plt.rc('ytick',labelsize=fS)
plt.rc('legend',fontsize=fS)
plt.rc('figure',titlesize=fS)




# %%= Load the head data ====================================================
import flopy.utils.binaryfile as bf
headobj = bf.HeadFile(modelName + ".hds")
times = headobj.get_times()
head = headobj.get_data(totim=times[-1])

idx = (0, int(nrow / 2) - 1, int(ncol / 2) )
ts = headobj.get_ts(idx)

#%%   =================  analytical solution =================
w1_x = 5000 + delr       # Location of pumping well x-coordinate
w1_y = 5000            # Location of pumping well y-coordinate
w1_outflow = -2000
xw = w1_x # x-location of well, m
yw = w1_y # y-location of well, m
Q = - w1_outflow # discharge of well, m^3/d positive for pumping
#Q = 1000
strt_left=-30; strt_right= -40
Lx   = 10000
U = (strt_left-strt_right)/Lx # flow gradient  [-]
nrow_analytical = nrow*10; 
ncol_analytical = ncol*10;
xg, yg = np.meshgrid(np.linspace(0, Lx, nrow_analytical), np.linspace(0, Lx, nrow_analytical))
Kh = 3
#theta_1 =  - U * xg +  Q / Kh / np.abs(botm-strt_right)/ (2 * np.pi) * np.log( np.sqrt((xg-xw)**2+(yg-yw)**2   )  )

botm_1=-80
theta_1 =  - U * xg +  Q / Kh / np.abs(botm_1-strt_right)/ (2 * np.pi) * np.log( np.sqrt((xg-xw)**2+(yg-yw)**2   )  )


# ===== Plot head in x direction ============================================
z_ore_top_m = - 35
z_ore_bot_m = - 39
x_ore_left_m = 5500
x_ore_right_m = 9000


fig = plt.figure(figsize=(16, 8),dpi=300)
ax = fig.add_subplot(1, 2, 1)
ax.plot([0,Lx],[strt_left,strt_right],'r:',label='Original watertable')


dx   = 100 # cell size in x-direction (along rows)
ax.plot(xg[int(ncol_analytical/2)]+dx, 
        (theta_1[int(ncol_analytical/2)] -theta_1[int(ncol_analytical/2)][-1] + strt_right ),
        'ro-',label='Analytical')
x, y, z = gwf.modelgrid.xyzcellcenters

ax.plot(x[1], head[0,int( nrow/2)+1,:],label='Numerical')

ax.plot([x_ore_left_m,x_ore_right_m,x_ore_right_m,x_ore_left_m,x_ore_left_m],
        [z_ore_bot_m,z_ore_bot_m,z_ore_top_m,z_ore_top_m,z_ore_bot_m],'r-',label = 'Ore' )
ax.set_title("Hydraulic Heads (m) ")
ax.set_xlabel("Horizontal distance (x-direction) [m]")
ax.set_ylabel("Groundwater Heads [m]")
ax.legend()
ax = fig.add_subplot(1, 2, 2)
ax.plot(ts[:, 0] / 365, ts[:, 1], "bo-")
ax.set_xlabel("Elapsed Time [years]", fontsize=fS)
ax.set_ylabel("Groundwater Heads [m]", fontsize=fS)
ax.set_title('Groundwater Head at the well')
plt.show()


#%% ====== plot map view =============
fig = plt.figure(figsize=(16, 9),dpi=300)
ax = fig.add_subplot(1, 2, 1, aspect="equal")
levels = np.linspace(1,10,10)
levels = np.linspace(-50,-30,9)
pmv = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=0)
arr = pmv.plot_array(head)
cs = pmv.contour_array(head, levels=levels, colors="black")
#ax.clabel(cs, fmt="%2.2f")
plt.colorbar(arr, shrink=0.5, ax=ax)
#pmv.plot_vector(qx, qy, istep=2, jstep=2, normalize=True, color="white")
qm = pmv.plot_ibound()
ax.set_title("Hydraulic Heads (Numerical) ")
ax.set_xlabel("Horizontal distance (x-direction) [m]")
ax.set_ylabel("Horizontal distance (y-direction) [m]")



ax = fig.add_subplot(1, 2, 2, aspect="equal")
levels = np.linspace(1,10,10)
levels = np.linspace(-50,-30,12)
a=ax.contourf(xg,yg,theta_1-theta_1[int(ncol_analytical/2)][-1] + strt_right,9)
b=ax.contour(xg,yg,theta_1-theta_1[int(ncol_analytical/2)][-1] + strt_right,9,colors='black')
plt.clabel(b, inline=True, fontsize=20)

ax.set_title("Hydraulic Heads (Analytical) ")
ax.set_xlabel("Horizontal distance (x-direction) [m]")
ax.set_ylabel("Horizontal distance (y-direction) [m]")
plt.colorbar(arr, shrink=0.5, ax=ax)
#ax.contour(xg,yg,theta_1, levels=levels, colors="black")
ax.clabel(cs, fmt="%2.2f")

plt.show()
# Summary of common palette of colors
# b = dark sky blue
# c = cyan
# g = leaf green
# k - black
# m = pink
# r = red
# y = pimple yellow
# w = white
# f9ee4a = pale yellow
# 44d9ff = pale blue
# f95b4a = orange

# Summary of common line types:
# '-' solid
# ':' dotted
# '--' dashed
# '-.' dash dot    
print('  ..... plotting complete.')
# =============================================================================
# ======== END OF SCRIPT ======== END OF SCRIPT ======== END OF SCRIPT ========
# =============================================================================

