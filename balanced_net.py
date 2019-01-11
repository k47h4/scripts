#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
recurrent network with plastic excitatory synapses, 
with triplet or STDP plasticity model
inspired by balanced network from 
Zenke, Hennequin, and Gerstner (2013) in PLoS Comput Biol 9, e1003330. 
doi:10.1371/journal.pcbi.1003330

Created on Wed Jan 2 14:12:15 2019

@author: kwilmes
"""


from brian2 import *
from brian2tools import *
from pypet import Environment, cartesian_product
import os, sys
import pandas as pd


defaultclock.dt = .1*ms
start_scope()

#np.random.seed(parameters.seed)

NE = 20000          # Number of excitatory inputs
NI = NE/4          # Number of inhibitory inputs
NP = 2500           # Number of exc. Poisson inputs

tau_e = 5.0*ms   # Excitatory synaptic time constant
tau_i = 10.0*ms  # Inhibitory synaptic time constant
tau_nmda = 100*ms # Excitatory NMDA time constant
alpha = .5       # ratio of AMPA and NMDA synapses
eta = 0

lambdae = 2*Hz     # Firing rate of Poisson inputs


tau_stdp = 20*ms    # STDP time constant
tau_stdp_slow = 40*ms # Triplet slow postsynaptic trace time constant

simtime = 1*second # Simulation time


gl = 10.0*nsiemens   # Leak conductance
el = -70*mV          # Resting potential
er = -80*mV          # Inhibitory reversal potential
vt = -50.*mV         # Spiking threshold

taum = 20*ms              # Excitatory membrane time constant
taum_i = 10*ms             # Inhibitory membrane time constant 

wPE_initial = 1.6   # (initial) excitatory synaptic weight from Poisson inputs
wEE_initial = 1.6   # initial recurrent excitatory weights
wEI_initial = 1.6   # (initial) weights from E to I
wIE_initial = 10.0  # (initial) weights from I to E
wII_initial = 10.0  # (initial) weights from I to I

gmax = 100               # Maximum inhibitory weight

Ap = 5e-4           # amplitude of LTP due to presynaptic trace
An = -Ap * 1.05     # amplitude of LTD due to fast postsynaptic trace
Ap2 = Ap            # optional different LTP amplitude for triplet term
An2 = An            # optional different LTD amplitude for triplet term
# if both Ap2 and An2 are zero: standard pair-based STDP

dApre = 1.
dApost = 1.
dApre2 = 1.
dApost2 = 1.


# In[3]:

eqs_inh_neurons='''
dv/dt=(-gl*(v-el)-((alpha*ge+(1-alpha)*g_nmda)*v+gi*(v-er)))*100*Mohm/taum_i : volt (unless refractory)
dg_nmda/dt = (-g_nmda+ge)/tau_nmda : siemens
dge/dt = -ge/tau_e : siemens
dgi/dt = -gi/tau_i : siemens
'''

eqs_exc_neurons='''
dv/dt=(-gl*(v-el)-((alpha*ge+(1-alpha)*g_nmda)*v+gi*(v-er)))*100*Mohm/taum : volt (unless refractory)
dge/dt = -ge/tau_e : siemens
dgi/dt = -gi/tau_i : siemens
dg_nmda/dt = (-g_nmda+ge)/tau_nmda : siemens
'''

inh_neurons = NeuronGroup(NI, model=eqs_inh_neurons, threshold='v > vt',
                      reset='v=el', refractory=8.3*ms, method='euler')
exc_neurons = NeuronGroup(NE, model=eqs_exc_neurons, threshold='v > vt',
                      reset='v=el', refractory=8.3*ms, method='euler')

inh_neurons.v = (10*np.random.rand(int(NI))-60)*mV
exc_neurons.v = (10*np.random.randn(NE)-60)*mV
# sample membrane potentials between -70 and -65 mV to prevent all neurons
# spiking at the same time initially

indep_Poisson = PoissonGroup(NP,lambdae)
    
connectionPE = Synapses(indep_Poisson, exc_neurons,
                on_pre='ge += wPE_initial*nS',
                name = 'PE')
connectionPE.connect(p=.05)


# In[4]:

eqs_tripletrule = '''w : 1
dApre/dt = -Apre / tau_stdp : 1 (event-driven)
dApost/dt = -Apost / tau_stdp : 1 (event-driven)
dApre2/dt = -Apre2 / tau_stdp_slow : 1 (event-driven)
dApost2/dt = -Apost2 / tau_stdp_slow : 1 (event-driven)'''

on_pre_triplet='''
Apre += dApre
Apre2before = Apre2
w = clip(w + eta * An * Apost, 0, gmax)
Apre2 += dApre2
ge += wEE_initial*nS'''
on_post_triplet='''
Apost += dApost
Apost2before = dApost2
w = clip(w + eta * Ap * Apre * Apost2before, 0, gmax)
Apost2 += dApost2'''

# optional different LTD amplitude for triplet term:
#w = clip(w + Apost * (An + An2 * Apre2before), 0, gmax)


eqs_stdp = '''
w : 1
dApre/dt=-Apre/tau_stdp : 1 (event-driven)
dApost/dt=-Apost/tau_stdp : 1 (event-driven)
'''
on_pre='''Apre += 1.
w = clip(w+Apost*eta*An, 0, gmax)
ge += w*nS'''
on_post='''Apost += 1.
w = clip(w+Apre*eta*Ap, 0, gmax)
'''

connectionEE = Synapses(exc_neurons, exc_neurons, model=eqs_tripletrule,
                on_pre = on_pre_triplet,
                on_post = on_post_triplet,
                name = 'EE')
connectionEE.connect(p=.05)
connectionEE.w = wEE_initial

connectionEI = Synapses(exc_neurons, inh_neurons,
                on_pre='ge += wEI_initial*nS',
                name = 'EI')
connectionEI.connect(p=.05)

connectionII = Synapses(inh_neurons, inh_neurons,
                on_pre='gi += wII_initial*nS',
                name = 'II')
connectionII.connect(p=.05)

connectionIE = Synapses(inh_neurons, exc_neurons,
                on_pre='gi += wIE_initial*nS',
                name = 'IE')
connectionIE.connect(p=.05)

# In[5]:

sm_Poiss = SpikeMonitor(indep_Poisson)

sm_inh = SpikeMonitor(inh_neurons)
vm_inh = StateMonitor(inh_neurons, 'v', record=[0])
sm_exc = SpikeMonitor(exc_neurons)
vm_exc = StateMonitor(exc_neurons, 'v', record=[0])
weight = StateMonitor(connectionEE, 'w', record=[0])


# In[6]:
net = Network(collect())
net.run(simtime)
BrianLogger.log_level_info()


plt.figure()
plot_raster(sm_inh.i, sm_inh.t, time_unit=second, marker=',', color='k')
plt.show()
plt.figure()
plot_raster(sm_exc.i, sm_exc.t, time_unit=second, marker=',', color='k')
plt.show()