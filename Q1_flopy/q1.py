import flopy
import numpy as np
import matplotlib.pyplot as plt


Lx = 100. # from plot, y is plotted from left to right
Ly = 10.  # from plot, y is plotted upward
ztop = 0.
zbot = -20.
#nlay = 10
nlay = 1
nrow = 1
ncol = 10
delr = Lx / ncol
delc = Ly / nrow
delv = (ztop - zbot) / nlay
botm = np.linspace(ztop, zbot, nlay + 1)

#dis.itmuni_dict
#dis.itmuni
hk = 30   # how to change the time unit in flopy?  ITMUNI in dis #dis.itmuni_dict
vka = 30
sy = 0.25
ss = 1.e-4
laytyp = 1   # unconfined



# Variables for the BAS package
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32) + 2
lay_id_chd=0
row_id_chd=0
col_id_chd=0
ibound_chd=1
ibound[lay_id_chd][row_id_chd][col_id_chd]=ibound_chd   # why do we need three layers here?? because the model is 3-d
#ibound[lay_id_chd][row_id_chd][1]=ibound_chd   # why do we need three layers here?? because the model is 3-d

strt = 10. * np.ones((nlay, nrow, ncol), dtype=np.float32)

# time step parameters
nper = 3
perlen =[100, 100, 100]  # in days i guess
nstp = [100, 100, 100]
steady = [True, False, False]

modelname = 'tutorial2'
mf = flopy.modflow.Modflow(modelname, exe_name='mf2005')
dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                                       top=ztop, botm=botm[1:],
                                      nper=nper, perlen=perlen, nstp=nstp, steady=steady,
                                      itmuni=4)
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vka, sy=sy, ss=ss, laytyp=laytyp, ipakcb=53)
pcg = flopy.modflow.ModflowPcg(mf)


# plot the domain
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
#modelmap = flopy.plot.ModelMap(sr=mf.dis.sr)
modelmap = flopy.plot.ModelMap(model=mf,rotation=0)  #suggested by flopy3_MapExample.ipynb
linecollection = modelmap.plot_grid()  # this line is the plotting line

linecollection = modelmap.plot_grid()  # this line is the plotting line
quadmesh = modelmap.plot_ibound()  # a useful command to check all the ibounds
fig.show()



#dis.sr.xcentergrid
#dis.sr.ycentergrid
#dis.sr.xgrid
#dis.sr.ygrid
#np.ma.masked_equal(dis.sr.ygrid,1000) # very useful command to find specific file locations
#modelmap.sr.vertices
#flopy.plot.plotutil.cell_value_points
#
#
#modelxsect = flopy.plot.ModelCrossSection(model=mf, line={'Row': 0})
#modelxsect.elev
#
#modelxsect.dis
#modelxsect.xpts
#modelxsect.xcentergrid
#modelxsect.zcentergrid

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# ibound nlay,nrow,ncol
#slicing a row for crosection plot

ib=ibound[:,0,:]

fig = plt.figure()
ax = fig.gca(projection='3d')
modelxsect = flopy.plot.ModelCrossSection(model=mf, line={'Row': 0})
surf = ax.plot_surface(modelxsect.xcentergrid, modelxsect.zcentergrid, ib, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show(block=False)


'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
'''

# we are supposed to use CHD boundary here not GHB boundary
'''
# Make list for stress period 1
stageleft = 10.
stageright = 10.
bound_sp1 = []
for il in range(nlay):
    condleft = hk * (stageleft - zbot) * delc
    condright = hk * (stageright - zbot) * delc
    for ir in range(nrow):
        bound_sp1.append([il, ir, 0, stageleft, condleft])
        bound_sp1.append([il, ir, ncol - 1, stageright, condright])
print('Adding ', len(bound_sp1), 'GHBs for stress period 1.')

# Make list for stress period 2
stageleft = 10.
stageright = 0.
condleft = hk * (stageleft - zbot) * delc
condright = hk * (stageright - zbot) * delc
bound_sp2 = []
for il in range(nlay):
    for ir in range(nrow):
        bound_sp2.append([il, ir, 0, stageleft, condleft])
        bound_sp2.append([il, ir, ncol - 1, stageright, condright])
print('Adding ', len(bound_sp2), 'GHBs for stress period 2.')

# We do not need to add a dictionary entry for stress period 3.
# Flopy will automatically take the list from stress period 2 and apply it
# to the end of the simulation, if necessary
stress_period_data = {0: bound_sp1, 1: bound_sp2}

# Create the flopy ghb object
ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=stress_period_data)
'''

#chd={0:[
#       [0,0,0,1,2],
#       [0,0,2,1,2],
#       ]
#    }

ibound_chd_mask=np.ma.masked_equal(ibound,ibound_chd)

chd_node_index=np.where(ibound_chd_mask.mask)
stress_period_data = {}

bound_sp0=[]
for i in np.arange(np.sum(ibound_chd_mask.mask)):
   bound_sp0.append([chd_node_index[0][i],chd_node_index[1][i],chd_node_index[2][i],0,0]  )
bound_sp1=[]
for i in np.arange(np.sum(ibound_chd_mask.mask)):
   bound_sp1.append([chd_node_index[0][i],chd_node_index[1][i],chd_node_index[2][i],-10,-10]  )

bound_sp2=[]
for i in np.arange(np.sum(ibound_chd_mask.mask)):
   bound_sp2.append([chd_node_index[0][i],chd_node_index[1][i],chd_node_index[2][i],0,0]  )



stress_period_data={0:bound_sp0,1:bound_sp1,2:bound_sp2}

chd=flopy.modflow.mfchd.ModflowChd(model=mf,stress_period_data=stress_period_data)


#flopy.modflow.mfchd.ModflowChd(model=mf,stress_period_data=chd)

stress_period_data = {}
for kper in range(nper):
    for kstp in range(nstp[kper]):
        if np.mod(kstp,50)==0:
            stress_period_data[(kper, kstp)] = ['save head',
                                                'save drawdown',
                                                'save budget',
                                                'print head',
                                                'print budget']
oc = flopy.modflow.ModflowOc(mf, stress_period_data=stress_period_data,compact=True)


mf.write_input()

# Run the model
success, mfoutput = mf.run_model(silent=True, pause=False, report=True)
if not success:
        raise Exception('MODFLOW did not terminate normally.')


# Imports
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf


# Create the headfile and budget file objects
headobj = bf.HeadFile(modelname+'.hds')
times = headobj.get_times()
cbb = bf.CellBudgetFile(modelname+'.cbc')


mytimes = [1.0, 101.0, 201.0]


fig = plt.figure(figsize=(8, 3))
ax = fig.add_subplot(1, 1, 1)
#modelxsect = flopy.plot.ModelCrossSection(model=mf, line={'Column': 5})  # this will only work when nrow is more than 1
modelxsect = flopy.plot.ModelCrossSection(model=mf, line={'Row': 0})
patches = modelxsect.plot_ibound()
linecollection = modelxsect.plot_grid()
t = ax.set_title('Row 0 Cross-Section with IBOUND Boundary Conditions')

#plot(dis.sr.xcentergrid.shape,head[0,0,:])
head = headobj.get_data(totim=mytimes[2])
ax.plot(dis.sr.xcentergrid[0,:],head[0,0,:])
head = headobj.get_data(totim=mytimes[1])
ax.plot(dis.sr.xcentergrid[0,:],head[0,0,:])

fig.show()





