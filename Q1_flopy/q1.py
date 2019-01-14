import flopy
import numpy as np
import matplotlib.pyplot as plt


Lx = 100. # from plot, y is plotted from left to right
Ly = 10.  # from plot, y is plotted upward
ztop = 0.
zbot = -20.
nlay = 10
#nlay = 1
nrow = 1
ncol = 10
delr = Lx / ncol
delc = Ly / nrow
delv = (ztop - zbot) / nlay
botm = np.linspace(ztop, zbot, nlay + 1)
hk = 0.06   # how to change the time unit in flopy?
vka = 0.06
sy = 0.25
ss = 1.e-4
laytyp = 1   # unconfined

# Variables for the BAS package
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32) + 2
ibound[0][0][0]=1   # why do we need three layers here??
strt = 10. * np.ones((nlay, nrow, ncol), dtype=np.float32)


# time step parameters
nper = 3
perlen = [1, 100, 100]
nstp = [1, 100, 100]
steady = [True, False, False]

modelname = 'tutorial2'
mf = flopy.modflow.Modflow(modelname, exe_name='mf2005')
dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                                       top=ztop, botm=botm[1:],
                                      nper=nper, perlen=perlen, nstp=nstp, steady=steady)
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


fig = plt.figure(figsize=(8, 3))
ax = fig.add_subplot(1, 1, 1)
#modelxsect = flopy.plot.ModelCrossSection(model=mf, line={'Column': 5})  # this will only work when nrow is more than 1
modelxsect = flopy.plot.ModelCrossSection(model=mf, line={'Row': 0})
patches = modelxsect.plot_ibound()
linecollection = modelxsect.plot_grid()
t = ax.set_title('Row 0 Cross-Section with IBOUND Boundary Conditions')
fig.show()

#dis.sr.xcentergrid
#dis.sr.ycentergrid
#dis.sr.xgrid
#dis.sr.ygrid
#np.ma.masked_equal(dis.sr.ygrid,1000) # very useful command
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




tress_period_data = {}
for kper in range(nper):
    for kstp in range(nstp[kper]):
        stress_period_data[(kper, kstp)] = ['save head',
                                            'save drawdown',
                                            'save budget',
                                            'print head',
                                            'print budget']
oc = flopy.modflow.ModflowOc(mf, stress_period_data=stress_period_data,
                             compact=True)




