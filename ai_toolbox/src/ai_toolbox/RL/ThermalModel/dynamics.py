#!/usr/bin/python3
# -*- coding: utf-8 -*-

#!pip install torch
import torch

###############################################################################################
###############################################################################################
# - Manu Lahariya, IDLab, 2/12/21
# Dynamics functions: functions based on physics of space heating
# - these functions will be used directly by the PhyCell class, which will be the recurrent unit
#   of the Thermal model
#
#
###############################################################################################
###############################################################################################

def BuildingT_next(RoomT, BuildingT, dt, Cb, Rb):
    '''
    :param RoomT: Room temperature at time T
    :param BuildingT: Building temperature at time T
    :param dt: Time difference in current and next time
    :param Cb: Capacitance of building thermal mass (Represents the storage capacity of building)
    :param Rb: Resistance between room and building thermal mass
    :return: Building temperature at timeslot T + dt
    '''
    return (dt/(Cb*Rb))*RoomT + (1 - (dt/(Rb*Cb)))* BuildingT


def BoilerInletT_next(RoomT ,BoilerInletT, dt, Ci, Ri):
    '''
    :param RoomT: Room temperature at time T
    :param BoilerInletT: Boiler Inlet temperature at time T
    :param dt: Time difference in current and next time
    :param Ci: Capacitance of boiler (Represents the storage capacity of boiler)
    :param Ri: Resistance between room and boiler
    :return: Building temperature at timeslot T + dt
    '''
    return (dt/(Ci*Ri))*RoomT + (1 - (dt/(Ci*Ri))) * BoilerInletT


def RoomT_next(RoomT, BuildingT, BoilerInletT,BoilerOutletT_next, AmbientT , dt, Cr, Ra, Rm, Ro, Ri,
               Irradiance = None):
    '''
    :param RoomT: Room temperature at time T
    :param BuildingT: Building temperature at time T
    :param BoilerInletT: Boiler Inlet temperature at time T
    :param BoilerOutletT: Boiler Outlet temperature at time T
    :param AmbientT: Ambient temperature at time T
    :param dt: Time difference in current and next time
    :param Cr: Capacitance of Room
    :param Cm: Capacitance of Building Thermal Mass
    :param Ci: Capacitance of boiler
    :param Ra: Resistance between room and ambient air
    :param Rm: Resistance between room and building thermal mass
    :param Ro: Resistance between room and boiler outlet temperature
    :param Ri: Resistance between room and boiler inlet temperature
    :return: Room temperature at timeslot T + dt
    '''
    Boiler_heating = (dt/(Cr*Ro))*BoilerOutletT_next
    Tr_contribution =  (1 - ((dt*Cr)*( 1/Ra + 1/Rm + 1/Ro + 1/Ri )))*RoomT
    Tm_contribution = (dt/(Rm*Cr))*BuildingT
    Ti_contribution = (dt / (Ri*Cr)) * BoilerInletT
    Ta_contribution = (dt/ (Ra*Cr )) * AmbientT
    if Irradiance is not None: Ta_contribution = Ta_contribution + Irradiance
    roomt_next =  Tr_contribution + Tm_contribution  + Ta_contribution + Boiler_heating + Ti_contribution
    return roomt_next

def BoilerOutletT_next(RoomT, BoilerSet_next,BoilerOutletT,dt, a1, a2):
    '''
    :param RoomT: Room temperature at time T
    :param BoilerSet_next: Boiler setpoint for the timestep T+dt
    :param BoilerOutletT: Boilter outlet temperature at time T
    :param dt: Time difference in current and next time
    :param a1: parameter a1
    :param a2: parameter a2
    :return:
    '''
    BoilerOutletT_next = (dt/a1) * unit_fun(BoilerSet_next)  * torch.relu(BoilerSet_next - BoilerOutletT) +\
                         (dt*a2) * (RoomT - BoilerOutletT) + \
                         (BoilerOutletT)
    return BoilerOutletT_next

def Gas_modulation( RoomT,
                     RoomSet_next,
                     BoilerInletT_next, BoilerOutletT,
                     BoilerOutletT_next,
                     BoilerSetpoint_next,
                     mdot, cg, b1,b2,
                     dt):
    # Modulation is limited between 0 and 100 - which represent the min and max modulation
    '''
    :param BoilerInletT: Boiler Inlet temperature at time T
    :param BoilerOutletT: Boiler Outlet temperature at time T
    :param mdot: Mass flow rate
    :param cg: specific heat capacity that is used to convert gas to temperature rise
    :return: Consumed Gas
    '''
    Gas = dt*mdot*cg*(BoilerOutletT_next - BoilerInletT_next) +\
          torch.abs(BoilerSetpoint_next - BoilerOutletT)/b1 -\
          b2*torch.abs(RoomSet_next - RoomT)
    Gas[Gas > 100] = 100.0
    Gas[Gas < 0.0] = 0.0
    return  Gas*unit_fun(BoilerSetpoint_next)


def unit_fun(x):
    unit_f = torch.clone(x)
    unit_f[unit_f > 0] = 1.0
    return unit_f








