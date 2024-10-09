# =============================================================================
# ===== CIVL4145 Modelling Project:
# ===== Base model (preliminary model) - Pumping from a unconfined aquifer
# ===== with uniform background flow.
# =====
#==============================================================================

#%% 0.0 ===== SETTING UP PYTHON ENVIRONMENT =====================================
print(' ')
print('  Importing packages and modules.....')
print(' ')
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import flopy
import flopy.utils.binaryfile as bf

# Check versions
print('  ..... system version: ',sys.version)
print('  ..... numpy version: {}'.format(np.__version__))
print('  ..... matplotlib version: {}'.format(mpl.__version__))
print('  ..... flopy version: {}'.format(flopy.__version__))
print('  ..... pakages and modules imported.')

#==============================================================================
# 1.0 ===== SETTING UP SIMULATION =============================================
#==============================================================================
print(' ')
print('  Setting up simulation.....')
#%% 1.1 ===== Set simulation and path details ===================================
modelName = 'SEAWAT-Pumping-Base' # Model name used when creating output files
SEAWAT_exe_Location = '../bin/swt_v4x64.exe'

# Create SEAWAT model object ('swt' becomes object)
swt = flopy.seawat.Seawat(modelName, exe_name=SEAWAT_exe_Location)
print('  ..... SEAWAT name file is: ',swt.namefile)
print('  ..... simulation object and paths set.')

#%% 1.2 ===== Model domain and grid definition ==================================
# Model domain and grid definition
dx   = 100 # cell size in x-direction (along rows)
dy   = 100 # cell size in y-direction (down columns)
Lx   = 10000 # Length in x-direction (use aquifer length -boundary cells added later)
Ly   = 10000 # Length in y-direction
ztop = 0.0 # Elevation of ground surface (can be an array)
# zbot = -9.0  # Elevation of base of aquifer (can be an array)
nlay = 1 # Number of model layers
nrow = int(Ly / dy)  # Number of rows (int function rounds to nearest integer)
ncol = int((Lx+2*dx)/dx)   # Number of cols (int function rounds to nearest integer)
# Note additional 2 columns added to model to be used as ocean boundary cells
# delv = (ztop - zbot) / nlay
# botm = np.linspace(ztop - delv, zbot, nlay)
botm = -80 #[-3.0, -10.0, -18.0] # Set elevation of each layer bottom (can be array)

# Time and length units
itmuni = 4 #Time units: 0=undef, 1=sec, 2=min, 3=hours, 4=days, 5=years, Default=4
nper   = 2 # Number of stress periods, Default=1
lenuni = 2 #length units: 0=undef, 1=feet, 2=m, 3=cm, Default=2
perlen = [50 *365 , 100*365 ] #length of stress period
nstp   = [50 , 50] #Number of timesteps in each stress period
steady = [False,False] 

# Create the discretization object and add to the MODFLOW model
dis = flopy.modflow.ModflowDis(swt, nlay, nrow, ncol, nper=nper, delr=dx,
                               delc=dy, laycbd=0, top=ztop,
                               botm=botm, itmuni=itmuni, lenuni=lenuni,
                               perlen=perlen, nstp=nstp,steady=steady)

# Full set of options (with default values):
# dis = flopy.modflow.mfdis.ModflowDis(model, nlay=1, nrow=2, ncol=2, nper=1,
#                               delr=1.0, delc=1.0, laycbd=0, top=1, botm=0,
#                               perlen=1, nstp=1, tsmult=1, steady=True,
#                               itmuni=4, lenuni=2, extension='dis',
#                               unitnumber=None, filenames=None, xul=None,
#                               yul=None, rotation=None, proj4_str=None,
#                               start_datetime=None)

# Note the Discretization File  is used to specify certain data used in all
# models. These include:
# 1.	the number of rows, columns and layers
# 2.	the cell sizes
# 3.	the presence of Quasi-3D confining beds
# 4.	the time discretization
#
# The Discretization File is required in all models.
print('  ..... model domain and grid set.')

#%% 1.3 ===== Boundary and initial conditions ===================================
# Boundary condition variables for the BAS package
ibound = np.zeros((nlay, nrow, ncol), dtype=np.float32) # Initialise variable
for k in range(nlay):
    ibound[k,:,:]  =  1 # Sets all cells to active
    ibound[k,:,0]  = -1 # Sets all west border (left) cells to constant head
    ibound[k,:,-1] = -1 # Sets all east border (right) cells to constant head

# Note an ibound value is specified for every cell in the model such that:
# if ibound < 0 the cell has constant head. A value of -1 is typically used.
# if ibound = 0 the cell is inactive (no flow cell). A value of 0 is used.
# if ibound > 0 the cell is active. A value of 1 is used.

# Initial condition variables for the BAS package
strt = np.zeros((nlay, nrow, ncol), dtype=np.float32) # Initialise variable
strt_left=-30; strt_right= -40
for k in range(nlay):
    strt[k,:,:]  = -35 # Sets inital hydraulic head values to 1.5 m
    strt[k,:,0]  = strt_left # Sets west (left) ocean boundary head to 0.0 m
    strt[k,:,-1] = strt_right # Sets east (right) ocean boundary head to 0.0 m
    
# Load initial hydraulic heads from CSV file
# (not used in this version but scripting retained in case you want to update
#  initial conditions with simulation data - make sure CSV data matches size
#  of model grid)
# infile_name = 'BaseModel-strt.csv' # Specify CSV file with data
# data = np.loadtxt(infile_name,delimiter=',') # Defines strt array using CSV
# strt = np.zeros((nlay, nrow, ncol), dtype=np.float32) # Initialise variable

# for k in range(nlay):
#     strt[k,:,:] = data
# del infile_name, data # Remove data from workspace    

# Create BAS object
bas = flopy.modflow.ModflowBas(swt, ibound=ibound, strt=strt) # Adds BAS package to MODFLOW

# Full set of options (with default values):
# bas = flopy.modflow.mfbas.ModflowBas(model, ibound=1, strt=1.0, ifrefm=True,
#                                     ixsec=False, ichflg=False, stoper=None,
#                                     hnoflo=-999.99, extension='bas',
#                                     unitnumber=None, filenames=None)

# Note the Basic package is used to specify certain data used in all models.
# These include:
# 1.	the locations of active, inactive, and specified head cells,
# 2.	the head stored in inactive cells, and
# 3.	the initial heads in all cells.
#
# The Basic package input file is required in all models.
print('  ..... boundary and initial conditions set in BAS package.')

#%% 1.4 ===== Adding flow packages ==============================================
# Note every model must  use one and only one of the three packages (BCF6, LPF,
# and HUF2) that are used to specify properties controlling flow between cells.

# Add Layer-Property Flow (LPF) package to the MODFLOW model

# Defining horizontal K values
#Kh = np.zeros((nlay, nrow, ncol), dtype=np.float32) # Initialise variable

#Kh[:,:,:] = 1      # Sets horizontal K values of layer 1
Kh = 3
#Kh[1,:,:] = 0.01  # Sets horizontal K values of layer 2
#Kh[2,:,:] = 11.0  # Sets horizontal K values of layer 3


# Define vertical K values
#Kv = Kh.copy()  # Copies Kh array to Kv (i.e., assumes same conditions in vertical)
Kv = Kh
# Specific yield
sy = np.zeros((nlay, nrow, ncol), dtype=np.float32) + 0.10 # Initialise variable
# sy[0,:,:] = 0.39  # Sets specific yeild values of layer 1
# sy[1,:,:] = 0.20  # Sets specific yeild values of layer 2
# sy[2,:,:] = 0.31  # Sets specific yeild values of layer 3

# # Updating layer 2 sy values for sand/indurated sand boundary
# Lay2_Change_x = 1000 + dx # Location of change from sand to indurated sand in Lay 2
# Lay2_Change_col = int(Lay2_Change_x / dx)+1
# sy[1,:,0:Lay2_Change_col] = 0.31

ss = 1.0e-5 # Specific storage

#laytyp = 1 # Sets layer type: 1 = uncofnined; 0 = confined
laytyp = 1 #[1,1,1] # Sets layer type: 1 = uncofnined; 0 = confined

# Create LPF object
ipakcb = 53 # save cell fluxes to unit 53
lpf = flopy.modflow.ModflowLpf(swt, hk=Kh, vka=Kv, laytyp=laytyp,
                               sy=sy, ss=ss, ipakcb=ipakcb)

# Full set of options (with default values):
# lpf = flopy.modflow.ModflowLpf(model, laytyp=0, layavg=0, chani=1.0,
#                               layvka=0, laywet=0, ipakcb=None, hdry=-1e+30,
#                               iwdflg=0, wetfct=0.1, iwetit=1, ihdwet=0,
#                               hk=1.0, hani=1.0, vka=1.0, ss=1e-05, sy=0.15,
#                               vkcb=0.0, wetdry=-0.01,
#                               storagecoefficient=False, constantcv=False,
#                               thickstrt=False, nocvcorrection=False,
#                               novfc=False, extension='lpf', unitnumber=None,
#                               filenames=None)
# Note the Layer-Property Flow package is used to specify properties
# controlling flow between cells.

# Create Block Centred Flow Pakcage (BCF) object
# Full set of options (with default values):
# bcf = flopy.modflow.mfbcf.ModflowBcf(model, ipabcf = kcb=None, intercellt=0,
#                               laycon=3, trpy=1.0, hdry=-1e+30, iwdflg=0,
#                               wetfct=0.1, iwetit=1, ihdwet=0, tran=1.0,
#                               hy=1.0, vcont=1.0, sf1=1e-05, sf2=0.15,
#                               wetdry=-0.01, extension='bcf', unitnumber=None,
#                               filenames=None)
# Note the Block-Centered Flow package is used to specify properties
# controlling flow between cells.
print('  ..... flow package set.')

#%% 1.6 ===== Well package =====================================================
# The Well package is used to simulate a specified flux to individual cells
# and specified in units of length^3 / time.

# Pumping well details
# Pumping well
w1_x = 5000 + dx       # Location of pumping well x-coordinate
w1_y = 5000            # Location of pumping well y-coordinate

w1_col = int(w1_x / dx)
w1_row = int(w1_y / dy)
w1_outflow = -2000 #-0.002*0.5*1000*1000  # [m3/day] depth * area (0.5 ha) #negaive value for pumping

# Initialising variables
itype = flopy.mt3d.Mt3dSsm.itype_dict()  #CZ ??
w_stress_period_data = {}
w1_sp1 = []

w1_sp1 = [nlay-1, w1_row, w1_col, 0]
w1_sp2 = [nlay-1, w1_row, w1_col, w1_outflow]


#Writing list variables for each well in each stress period
# for k in range(nlay):
#     w1_sp1.append([k, w1_row, w1_col, w1_outflow/nlay]) # lay, row, col index, pumping rate

# w1_sp1.append([0, w1_row, w1_col, w1_outflow/2])
# w1_sp1.append([nlay-1, w1_row, w1_col, w1_outflow/2])

# define well stress period {period, well info dictionary}
w_stress_period_data = {0: w1_sp1,1:w1_sp2}

# Note stress period data has the genreal form:
# stress_period_data =
#  {0: [[lay, row, col, flux], [lay, row, col, flux], [lay, row, col, flux] ],
#   1: [[lay, row, col, flux], [lay, row, col, flux], [lay, row, col, flux] ], ...
#   kper: [[lay, row, col, flux], [lay, row, col, flux], [lay, row, col, flux]]}

##stress_period_data = {0: wel_sp1}

# Add the well package
wel = flopy.modflow.ModflowWel(swt,
                               stress_period_data=w_stress_period_data,
                               ipakcb=53)

# Full set of options (with default values):
# flopy.modflow.mfwel.ModflowWel(model, ipakcb=None,
#                               stress_period_data=None,
#                               dtype=None, extension='wel', options=None,
#                               binary=False, unitnumber=None, filenames=None)
print('  ..... well package defined.')

#%% 1.6 ===== Output Control ====================================================
# Add OC package to the MODFLOW model

sp_data = {}
for kper in range(nper):
    for kstp in range(nstp[kper]):
        sp_data[(kper, kstp)] = ['print head', 'print budget', 'save head',
                                 'save budget']

# spd_data = {(0, 0): ['print head', 'print budget', 'save head', 'save budget']}

# Create Output Control (OC) package
oc = flopy.modflow.ModflowOc(swt, stress_period_data=sp_data, compact=True)

# Full set of options (with default values):
# oc = flopy.modflow.ModflowOc(model, ihedfm=0, iddnfm=0, chedfm=None,
#                           cddnfm=None, cboufm=None, compact=True,
#                           stress_period_data={(0, 0): ['save head']},
#                           extension=['oc', 'hds', 'ddn', 'cbc', 'ibo'],
#                           unitnumber=None, filenames=None, label='LABEL',
#                           **kwargs)
#
# Note: There are several ways of controlling the output generated by MODFLOW.
# The Output Control Option allows the user to specify when when heads,
# drawdown, water budget, and IBOUND are printed in the listing file or saved
# to an external file. If the Output Control Option is not used, head and
# overall budget are written to the listing file (printed) at the end of every
# stress period. For transient models, the size of the output files generated
# by using the Output Control Option can become very large.
print('  ..... output controls set.')  

#%% 1.7 ===== Setting Solver Options ============================================
# Every MODFLOW model must include one and only one of the solver packages.
# The Preconditioned Conjugate-Gradient (PCG) solver is very widely used.
# The PCG package is used to solve the finite difference equations in each
# step of a MODFLOW stress period.

# Add PCG package to the MODFLOW model
pcg = flopy.modflow.ModflowPcg(swt, hclose=1.0e-5)

# Full set of options (with default values):
# pcg = flopy.modflow.mfpcg.ModflowPcg(model, mxiter=50, iter1=30, npcond=1,
#                               hclose=1e-05, rclose=1e-05, relax=1.0,
#                               nbpol=0, iprpcg=0, mutpcg=3, damp=1.0,
#                               dampt=1.0, ihcofadd=0, extension='pcg',
#                               unitnumber=None, filenames=None)
#
print('  ..... solver options set.') 

# Write the input files
swt.write_input()

print('  ..... SEAWAT structure prepared')

#==============================================================================
# 4.0 ===== RUN THE SEAWAT MODEL ==============================================
#==============================================================================
print(' ')
print('  Running the simulation.....')
success, buff = swt.run_model(silent=False, report=True)
if not success:
    raise Exception("SEAWAT did not terminate normally.")

print(' ')
print('  ..... SEAWAT simulation complete')

#==============================================================================
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

#%% 5.2 ===== Load data from simulation output files ===========================
# tIDX = -1  # Sets time step to extract. 0 = 1st step, -1 = last step
# #===== Load salt concentration data
# ucnobj = bf.UcnFile("MT3D001.UCN", model=swt) # Creates object that reads file
# times = ucnobj.get_times() # Generates a list of output times
# times_years = [ii/365 for ii in times] # Converts time from days to years
# C_salt = ucnobj.get_data(totim=times[tIDX]) # Gets dat at last timestep

# #===== Load contaminant concentration data (Instantaneous source)
# ucnobj2 = bf.UcnFile("MT3D002.UCN", model=swt) # Creates object that reads file
# C_cont2 = ucnobj2.get_data(totim=times[tIDX]) # Gets dat at last timestep

#==== Extracting all concentration data to separate variable (for breakthough)
# #C2_all = ucnobj2.get_alldata()

# # Defining observation point location
# # Point 1
# ob1_x = 3400 + dx
# ob1_y = Ly - 1900
# ob1_col = int(ob1_x / dx)+1
# ob1_row = int(ob1_y / dy)

# #Point 2
# ob2_x = 4500 + dx
# ob2_y = Ly - 1600
# ob2_col = int(ob2_x / dx)+1
# ob2_row = int(ob2_y / dy)

# # Extracting concentration data from observation point cell
# # Note: Format of concentration data array is [time, layer, row, column]
# spill_ly0_c2data = C2_all[:,0,spill_row,spill_col]   # Spill site
# w1_ly0_c2data = C2_all[:,0,w1_row,w1_col]            # Pumping well
# ob1_ly0_c2data = C2_all[:,0,ob1_row,ob1_col]         # Observation point 1
# ob2_ly0_c2data = C2_all[:,0,ob2_row,ob2_col]         # Observation point 2

# # Definign irrigation area for plotting
# irr_x = [pine_east_x, pine_east_x, pine_west_x, pine_west_x, pine_east_x]
# irr_y = [Ly-pine_south_y, Ly-pine_north_y, Ly-pine_north_y, \
#          Ly-pine_south_y, Ly-pine_south_y]

#%% ===== Load the head data ====================================================
headobj = bf.HeadFile(modelName + ".hds")
times = headobj.get_times()
head = headobj.get_data(totim=times[-1])

idx = (0, int(nrow / 2) - 1, int(ncol / 2) )
ts = headobj.get_ts(idx)

#%%   =================  analytical solution =================
xw = w1_x # x-location of well, m
yw = w1_y # y-location of well, m
Q = - w1_outflow # discharge of well, m^3/d positive for pumping
#Q = 1000
U = (strt_left-strt_right)/Lx # flow gradient  [-]
nrow_analytical = nrow*10; 
ncol_analytical = ncol*10;
xg, yg = np.meshgrid(np.linspace(0, Lx, nrow_analytical), np.linspace(0, Lx, nrow_analytical))

theta_1 =  - U * xg +  Q / Kh / np.abs(botm-strt_right)/ (2 * np.pi) * np.log( np.sqrt((xg-xw)**2+(yg-yw)**2   )  )


# ===== Plot head in x direction ============================================
z_ore_top_m = - 35
z_ore_bot_m = - 39
x_ore_left_m = 5500
x_ore_right_m = 9000


fig = plt.figure(figsize=(16, 8),dpi=300)
ax = fig.add_subplot(1, 2, 1)
ax.plot([0,Lx],[strt_left,strt_right],'r:',label='Original watertable')

ax.plot(xg[int(ncol_analytical/2)]+dx, 
        (theta_1[int(ncol_analytical/2)] -theta_1[int(ncol_analytical/2)][-1] + strt_right ),
        'ro-',label='Analytical')

ax.plot(dis.get_node_coordinates()[1], head[0,int( nrow/2)+1,:],label='Numerical')

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
fig.show()
#%% ====== plot map view =============
fig = plt.figure(figsize=(16, 9),dpi=300)
ax = fig.add_subplot(1, 2, 1, aspect="equal")
levels = np.linspace(1,10,10)
levels = np.linspace(-50,-30,9)
pmv = flopy.plot.PlotMapView(model=swt, ax=ax, layer=0)
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