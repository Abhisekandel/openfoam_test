import math
import bisect
import os
import numpy as np
import matplotlib.pyplot as plt
import re

cwd = os.getcwd() # returns current working directory of a process

#cwd = cwd[:-7]

#--- Input Data ------------------------------
#tInit = 0.0087266    # set initial time
#tEnd = 3    # set end time


bladeCount = 4


R = 0.085       #radius of rotation
Chord = 0.0567
deltaZ = 0.013608   #blade thickness
b = 2 * 0.0425

#rho = 1.1436
#nu = 1.66*10**(-5)
#omega = 555.0147         # set current rotation speed -> dynamicMeshDict


with open(cwd + "/0/include/initialConditions", "r") as initialCond:
    initLines = initialCond.readlines()[7:]

    for iLine in range(len(initLines)):
        if re.search(r"RPM ", initLines[iLine]):
            RPM = float((re.split(r'\s', initLines[iLine])[-2]).replace(';',''))
        if re.search(r"nu ", initLines[iLine]):
            nu = float((re.split(r'\s', initLines[iLine])[-2]).replace(';',''))
        if re.search(r"rho0 ", initLines[iLine]):
            rho = float((re.split(r'\s', initLines[iLine])[-2]).replace(';',''))
#--- end Input Data ------------------------------


omega = 2*math.pi*RPM/60
c = re.search(r"([A-Z]\d{3}(_\d)?)/pimpleFoam",cwd)
caseName = c.group(1)

Reynolds = omega*R*Chord/nu

Fx, Fy, Fz, t, Mx, My, Mz = [],[],[],[],[],[],[] # define empty vectors

Fxges, Fyges, Fzges,tges, Mxges, Myges, Mzges = [],[],[],[],[],[],[]


iBlades = range(0,bladeCount)
#iBlades = [0]

#--- Extract Data from files (postProcessing) ------------------------------
caselist = os.listdir(cwd+"/postProcessing/airfoilUp0")
caselist = sorted(caselist)
caselist = caselist[1:]
for case in caselist:
    Fx, Fy, Fz, t, Mx, My, Mz = [],[],[],[],[],[],[]
    for blade in iBlades:
        for sides in ["Down", "Up"]:        # for each side of the airfoil
                with open(cwd+"/postProcessing/airfoil"+sides+str(blade)+"/"+case+"/force.dat", "rt")    as f:    # open the FORCES file of the airfoil
                    lines = f.readlines()[4:]        # read lines at line numbre 4 to end
                    t_tmp = [float(line.split()[0]) for line in lines]        # extract first column, convert to float
                    Fx_tmp = [float(line.split()[1].replace('(','')) for line in lines]        # extract second column, convert to flaot, replace the "(" with "" = nothing
                    Fy_tmp = [float(line.split()[2]) for line in lines]        # exctract third column, convert to float
                    Fz_tmp = [float(line.split()[3].replace(')','')) for line in lines]        # exctract third column, convert to float
                
                    if not Fy:
                        Fx = Fx_tmp
                        Fy = Fy_tmp
                        Fz = Fz_tmp
                    else:
                         Fx = [x+y for x,y in zip(Fx_tmp,Fx)] # add to the existing force the new one, zip generates a tuple 
                         Fy = [x+y for x,y in zip(Fy_tmp,Fy)]
                         Fz = [x+y for x,y in zip(Fz_tmp,Fz)]
                
                    if not t:
                        t = t_tmp
                    elif not t == t_tmp:
                        print("WARNING: times sets are DIFFERENT!!")
            
                with open(cwd+"/postProcessing/airfoil"+sides+str(blade)+"/"+case+"/moment.dat", "rt") as f: # open the MOMENT file of the airfoil
                    lines = f.readlines()[4:] # read lines at line numbre 4 to end
    #                t_tmp = [float(line.split()[0]) for line in lines]
                    Mx_tmp = [float(line.split()[1].replace('(','')) for line in lines]
                    My_tmp = [float(line.split()[2]) for line in lines]
                    Mz_tmp = [float(line.split()[3].replace(')','')) for line in lines]
                
                    if not Mz:
                        Mx = Mx_tmp
                        My = My_tmp
                        Mz = Mz_tmp
                    else:
                        Mx = [x+y for x,y in zip(Mx_tmp,Mx)]
                        My = [x+y for x,y in zip(My_tmp,My)]
                        Mz = [x+y for x,y in zip(Mz_tmp,Mz)]
    for q in range(len(Fx)):
        tges.append(t[q])
        Fxges.append(Fx[q])
        Fyges.append(Fy[q])
        Fzges.append(Fz[q])
        Mxges.append(Mx[q])
        Myges.append(My[q])
        Mzges.append(Mz[q])
        
#--- end Extract Data ------------------------------

j = 0

Tges = [math.sqrt(x**2+y**2) for x,y in zip(Fxges,Fyges)] 
nRot=int(tges[-1]*omega/(2*math.pi))      # number of complete rotaions of case

# -- Insert start position ----------------------------------------------------
# tges.insert(0, 0)
# Fxges.insert(0, 0)
# Fyges.insert(0, 0)
# Mzges.insert(0, 0)

factor=(2*-omega*(rho*R*b)**(0.5))**(-1)
tFixed, FxMean, FyMean, TMean, MzMean, FOMMean, AMean = [], [], [], [], [], [], []
futureStart = 0

matFx, matFy, matT, matMz, matTime= [], [], [], [],[]

for i in np.arange(1, nRot+1, 1):
    vecFx, vecFy, vecT, vecMz, vecTime = [], [], [], [],[]

    while tges[j]*omega/(2*math.pi) < i:
        vecFx.append(Fxges[j])
        vecFy.append(Fyges[j])
        vecMz.append(Mzges[j])
        vecT.append(Tges[j])
        vecTime.append(tges[j])
        j +=1
    
    currentStart = futureStart
    futureStart= currentStart + len(vecFx)

    tFixed  = np.linspace(tges[currentStart], tges[futureStart], num=3601, endpoint=True) # creates an arry with 3600 points between tInit and tEnd
    tFixed = tFixed[:-1]
    FxMean.append(2*np.mean(np.interp(tFixed, tges, Fxges)))
    FyMean.append(2*np.mean(np.interp(tFixed, tges, Fyges)))
    TMean.append(math.sqrt(FyMean[i-1]**2+FxMean[i-1]**2))
    MzMean.append(2*np.mean(np.interp(tFixed, tges, Mzges)))
    FOMMean.append(-1*TMean[i-1]**(3/2)/abs(MzMean[i-1])*factor)
    AMean.append(math.degrees(math.atan2(FyMean[i-1],FxMean[i-1])))
    
    for k in range(len(vecTime)):
        vecTime[k] = vecTime[k] - (i-1) * 2 *math.pi/omega
        vecTime[k] = vecTime[k]*360*omega/(2*math.pi)
    
    matFx.append(vecFx)
    matFy.append(vecFy)
    matT.append(vecT)
    matMz.append(vecMz)
    matTime.append(vecTime)





#tInit=tges[1000]
tEnd = tges[-1]#-2*2*math.pi/(omega)
tInit = tEnd-1*2*math.pi/(omega)


#--- Cut-Out Data within Range ------------------------------
cutI=bisect.bisect(tges,tInit)#13*2*math.pi/(omega))#tInit)         # get position of tInit inside of the tim-list t
cutE=bisect.bisect(tges,tEnd)#14*2*math.pi/(omega))#tEnd)     # get position of tEnd inside of the tim-list t
tges=tges[cutI:cutE+1]        # extract the time between tInit and tEnd and set as new time-list t
Fyges=Fyges[cutI:cutE+1]        # extract the force between tInit and tEnd
Fxges=Fxges[cutI:cutE+1]        # see above
Fzges=Fzges[cutI:cutE+1]
Mxges=Mxges[cutI:cutE+1]
Myges=Myges[cutI:cutE+1]
Mzges=Mzges[cutI:cutE+1]
Tges = [math.sqrt(x**2+y**2) for x,y in zip(Fxges,Fyges)]    
#--- end Cut-Out Data ------------------------------

if tInit * (omega/2/math.pi)%1 ==0:
    rot = [360*(x-tInit)*(omega/2/math.pi) for x in tges]
else:
    subRot = int(tInit * (omega/2/math.pi)/1 +1)
    rot = [360*(x*(omega/2/math.pi)-subRot) for x in tges]


#--- Calcuate Data ------------------------------
P = [x*-2*omega for x in Mzges]        # calculates the power: P=M * \omega
E = [math.sqrt(x**2+y**2)/p for (p,x,y) in zip(P,Fxges,Fyges)]        # calculates the energy: E = F.res / Power
tFixed0  = np.linspace(tInit, tEnd, num=3601, endpoint=True)        # creates an arry with 3600 points between tInit and tEnd
tFixed0 = tFixed0[:-1]
FyMean0 = 2*np.mean(np.interp(tFixed0, tges, Fyges))        # first interpolates the values of Fy between the given points ini tFixed, calculates the mean value of the interpolated array
FxMean0 = 2*np.mean(np.interp(tFixed0, tges, Fxges))        # see above
FzMean0 = np.mean(np.interp(tFixed0, tges, Fzges))        # see above
TMean0 = math.sqrt(FyMean0**2+FxMean0**2)        # mean value of total force
T_angle0 = math.degrees(math.atan2(FyMean0,FxMean0))
MxMean0 = np.mean(np.interp(tFixed0, tges, Mxges))
MyMean0 = np.mean(np.interp(tFixed0, tges, Myges))
MzMean0 = 2*np.mean(np.interp(tFixed0, tges, Mzges))
PMean0 = np.mean(np.interp(tFixed0, tges, P))
EMean0 = TMean0/PMean0

# Figure Of Merit: F.O.M. Doudou
#factor2=(2*omega*(rho*R*deltaZ)**(0.5))**(-1)
#FOMMean2 = TMean**(3/2)/MzMean*factor2

# TMean = 3.5525
# MzMean = -59.965/omega

#FOMCorr

FOMMean0 = TMean0**(3/2)/MzMean0*factor

#--- end Calcuate Data ------------------------------

print('Input Data:'+
      '\nCase: '+caseName+
      '\nForces and Moments for '+ str(len(iBlades))+' Blades'+
      '\nRPM = %0.1f 1/min' %RPM+
      '\nOmega = %0.2f rad/s' %omega+
      '\nRe = %.2e' %Reynolds+
'\n\nMeans for current case (Applied Symetry):' +
'\nFy= %0.4f' %FyMean0 + ' N; ' +
'\nFx= %0.4f' %FxMean0 + ' N; ' +
'\nT= %0.4f' %TMean0 + ' N; ' +
'\nT angle= %0.4f' %T_angle0 + ' N; ' +
'\nMz= %0.4f' %MzMean0 + ' Nm; ' +
'\nP= %0.4f' %PMean0 + ' W; ' +
'\nE= %0.4f' %EMean0 + ' J; ' + 
'\n\nFOM= %0.4f' %FOMMean0 + '; '+
'\n\n{Absolute values of cancelled out forces and moments}' +
'\n{Fz= %0.4f' %FzMean0 + ' N}; ' +
'\n{Mx= %0.4f' %MxMean0 + ' Nm}; ' +
'\n{My= %0.4f' %MyMean0 + ' Nm}; ')

lines = ['Input Data:',
         '\nCase: '+caseName,
         '\nForces and Moments for '+ str(len(iBlades))+' Blades',
         '\nRPM = %0.1f 1/min' %RPM,
         '\nOmega = %0.2f rad/s' %omega,
         '\nRe = %.2e' %Reynolds,
'\nMeans for current case (Applied Symetry):',
'\nFy= %0.4f' %FyMean0 + ' N; ',
'\nFx= %0.4f' %FxMean0 + ' N; ',
'\nT= %0.4f' %TMean0 + ' N; ',
'\nT angle= %0.4f' %T_angle0 + ' N; ',
'\nMz= %0.4f' %MzMean0 + ' Nm; ',
'\nP= %0.4f' %PMean0 + ' W; ',
'\nE= %0.4f' %EMean0 + ' J; ',
'\n\nFOM= %0.4f' %FOMMean0 + '; ',
'\n\n{Absolute values of cancelled out forces and moments}',
'\n{Fz= %0.4f' %FzMean0 + ' N}; ',
'\n{Mx= %0.4f' %MxMean0 + ' Nm}; ',
'\n{My= %0.4f' %MyMean0 + ' Nm}; ']

with open(cwd+'/postProcessing/ForceAndPowerMeans.txt', 'w') as f:
    f.write('\n'.join(lines))
    
#fig =plt.figure(figsize=(32, 18), dpi=100)  # Create a figure containing a single axes.
#plt.plot(t,Mz)  # Plot some data on the axes.

# Major ticks every 20, minor ticks every 5
majorRange = range(0,int(max(rot)+5),45)
major_ticks = []
for majorEntry in majorRange:
    major_ticks.append(majorEntry)

if min(rot)<0:
    if abs(min(rot))%45>25:
        arrstart = int(min(rot)+(abs(min(rot))%45)-45)
    else:
        arrstart = int(min(rot)+(abs(min(rot))%45))
    majTicks = range(arrstart,0,45)
    for majEntry in majTicks:
        major_ticks.append(majEntry)

    



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(32, 18), dpi=100) #, sharex=True, sharey=True)
fig.suptitle('Case: '+caseName+' - Forces and Moments for '+ str(len(iBlades))+' Blades\n'+'Omega = %0.2f rad/s' %omega+' (Re = %.2e)' %Reynolds)
ax1.plot(rot, Fyges)
ax1.set_title('Force Y')
ax1.set(ylabel='Fy [N]')
ax1.set(xlabel='Psi in Degree')
ax1.axhline(y=0, color='k')
ax1.grid(visible=True)
ax1.set_xticks(major_ticks)
ax2.plot(rot, Fxges, 'tab:orange')
ax2.set_title('Force X')
ax2.set(ylabel='Fx [N]')
ax2.set(xlabel='Psi in Degree')
ax2.axhline(y=0, color='k')
ax2.grid(visible=True)
ax2.set_xticks(major_ticks)
ax3.plot(rot, Mzges, 'tab:green')
ax3.set_title('Moment Z')
ax3.set(ylabel='Mz [Nm]')
ax3.set(xlabel='Psi in Degree')
ax3.axhline(y=0, color='k')
ax3.grid(visible=True)
ax3.set_xticks(major_ticks)
ax4.plot(rot, Tges, 'tab:red')
ax4.set_title('resulting Force')
ax4.set(ylabel='F total [N]')
ax4.set(xlabel='Psi in Degree')
ax4.axhline(y=0, color='k')
ax4.grid(visible=True)
ax4.set_xticks(major_ticks)

plt.show()

fig2, ((ax5, ax6, ax7), (ax8, ax9, ax10)) = plt.subplots(2, 3,figsize=(32, 18), dpi=100) #, sharex=True, sharey=True)
fig2.suptitle('Case: '+caseName+' - Forces and Moments for '+ str(len(iBlades))+' Blades\n'+'Omega = %0.2f rad/s' %omega+' (Re = %.2e)' %Reynolds)
ax5.plot(rot, Fyges)
ax5.set_title('Force Y')
ax5.set(ylabel='Fy [N]')
ax5.set(xlabel='Psi in Degree')
ax5.axhline(y=0, color='k')
ax5.grid(visible=True)
ax5.set_xticks(major_ticks)
ax6.plot(rot, Fxges, 'tab:orange')
ax6.set_title('Force X')
ax6.set(ylabel='Fx [N]')
ax6.set(xlabel='Psi in Degree')
ax6.axhline(y=0, color='k')
ax6.grid(visible=True)
ax6.set_xticks(major_ticks)
ax7.plot(rot, Fzges, 'tab:cyan')
ax7.set_title('Force Z')
ax7.set(ylabel='Fz [N]')
ax7.set(xlabel='Psi in Degree')
ax7.axhline(y=0, color='k')
ax7.grid(visible=True)
ax7.set_xticks(major_ticks)
ax8.plot(rot, Myges, 'tab:red')
ax8.set_title('Moment Y')
ax8.set(ylabel='My [Nm]')
ax8.set(xlabel='Psi in Degree')
ax8.axhline(y=0, color='k')
ax8.grid(visible=True)
ax8.set_xticks(major_ticks)
ax9.plot(rot, Mxges, 'tab:pink')
ax9.set_title('Moment X')
ax9.set(ylabel='Mx [Nm]')
ax9.set(xlabel='Psi in Degree')
ax9.axhline(y=0, color='k')
ax9.grid(visible=True)
ax9.set_xticks(major_ticks)
ax10.plot(rot, Mzges, 'tab:green')
ax10.set_title('Moment Z')
ax10.set(ylabel='Mz [Nm]')
ax10.set(xlabel='Psi in Degree')
ax10.axhline(y=0, color='k')
ax10.grid(visible=True)
ax10.set_xticks(major_ticks)

plt.show()

#---  Plot Data  -----------------------------------------------------------
fig, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2,figsize=(32, 18), dpi=100) 
fig.suptitle("Postprocessing:   " + cwd)

major_ticks = np.arange(0, 370, 45)
StartRot=1     # at which rotation will start the plot

#--- ax1 -------------------------------------------------
for i in np.arange(StartRot, nRot, 1):
    ax11.plot(matTime[i], matFy[i], label="Rot " + str(i))#, 'green')

ax11.set_title("Fy")
ax11.set(ylabel="Fy in [N]")
ax11.set(xlabel="Psi in degree")
ax11.grid()
ax11.set_xticks(major_ticks)
ax11.set_xlim(0, 360)
ax11.axhline(y=0, color='k')
ax11.legend(loc="upper right")
#--- ax2 -------------------------------------------------
for i in np.arange(StartRot, nRot, 1):
    ax12.plot(matTime[i], matFx[i], label="Rot " + str(i))#'orange')

ax12.set_title("Fx")
ax12.set(ylabel="Fx in [N]")
ax12.set(xlabel="Psi in degree")
ax12.set_xticks(major_ticks)
ax12.grid()
ax12.set_xlim(0, 360)
ax12.axhline(y=0, color='k')
ax12.legend(loc="upper right")
#--- ax3 -------------------------------------------------
for i in np.arange(StartRot, nRot, 1):
    ax13.plot(matTime[i], matT[i], label="Rot " + str(i))#'b')
    
ax13.set_title("Total Thrust")
ax13.set(ylabel="Total Thrust in [N]")
ax13.set(xlabel="Psi in degree")
ax13.grid()
ax13.set_xticks(major_ticks)
ax13.set_xlim(0, 360)
ax13.axhline(y=0, color='k')
ax13.legend(loc="upper right")
#--- ax4 -------------------------------------------------
for i in np.arange(StartRot, nRot, 1):
    ax14.plot(matTime[i], matMz[i], label="Rot " + str(i))#'r')

ax14.set_title("Moment Z")
ax14.set(ylabel="Mz in [Nm]")
ax14.set(xlabel="Psi in degree")
ax14.set_xticks(major_ticks)
ax14.grid()
ax14.set_xlim(0, 360)
ax14.axhline(y=0, color='k')
ax14.legend(loc="upper right")
#---  end: Plot Data  -----------------------------------------------------------


# #---  Plot Mean  -----------------------------------------------------------
fig, ((ax15, ax16), (ax17, ax18)) = plt.subplots(2, 2,figsize=(32, 18), dpi=100) 
fig.suptitle("Postprocessing:   " + cwd)
Rotation = range(1, nRot+1, 1)

#--- ax1 -------------------------------------------------
ax15.plot(Rotation[StartRot:], TMean[StartRot:], 'green')
ax15.set_title("Mean Thrust")
ax15.set(ylabel=" Thrust in [N]")
ax15.set(xlabel="Psi in degree")
ax15.grid()
ax15.axhline(y=0, color='k')
#--- ax2 -------------------------------------------------
ax16.plot(Rotation[StartRot:], MzMean[StartRot:], 'orange')
ax16.set_title("Mean moment Z")
ax16.set(ylabel="Mz in [Nm]")
ax16.set(xlabel="Rotaton")
ax16.grid()
ax16.axhline(y=0, color='k')
#--- ax3 -------------------------------------------------
ax17.plot(Rotation[StartRot:], FOMMean[StartRot:], 'r')
ax17.set_title("Figure of Merit")
ax17.set(ylabel="FOM in [-]")
ax17.set(xlabel="Rotaton")
ax17.grid()
ax17.axhline(y=0, color='k')
#--- ax4 -------------------------------------------------
ax18.plot(Rotation[StartRot:], AMean[StartRot:], 'blue')
ax18.set_title("Angle of Thrust")
ax18.set(ylabel="Angle in degree")
ax18.set(xlabel="Rotaton")
ax18.grid()
ax18.axhline(y=0, color='k')
#---  end: Plot Data  -----------------------------------------------------------