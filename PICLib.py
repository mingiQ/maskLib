# -*- coding: utf-8 -*-
"""
Created on Sat 8/9/2025

@author: Mingi

Library for drawing standard Photonic IC design for hunglab -- based on slab masklib

"""

import maskLib.MaskLib as m
from dxfwrite import DXFEngine as dxf
from dxfwrite import const
from dxfwrite.entities import Polyline, Solid
from dxfwrite.vector2d import vadd, midpoint ,vsub, vector2angle, magnitude, distance
from dxfwrite.algebra import rotate_2d
from dxfwrite.mixins import SubscriptAttributes
from dxfwrite.base import DXFList,dxfstr

from maskLib.Entities import SolidPline, SkewRect, CurveRect, RoundRect, InsideCurve
from maskLib.utilities import kwargStrip, curveAB

from copy import deepcopy
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from copy import copy

import math as ma
from scipy import integrate



def alignmarker_gen(marker0, ebeam_writefield, markersize=10, offset_x=0, offset_y=0):
    X = ebeam_writefield/2+offset_x
    Y = ebeam_writefield/2+offset_y
    marker1 = [tuple(map(lambda a,b: a+b, marker0, (X,Y))), 
               tuple(map(lambda a,b: a+b, marker0, (X+markersize, Y+markersize))), 
               tuple(map(lambda a,b: a+b, marker0, (X+markersize/2, Y+markersize/2)))]
    marker2 = [tuple(map(lambda a,b: a+b, marker0, (-X,Y))), 
               tuple(map(lambda a,b: a+b, marker0, (-X-markersize, Y+markersize))),  
               tuple(map(lambda a,b: a+b, marker0, (-X-markersize/2, Y+markersize/2)))]
    marker3 = [tuple(map(lambda a,b: a+b, marker0, (X, -Y))), 
               tuple(map(lambda a,b: a+b, marker0, (X+markersize, -Y-markersize))), 
               tuple(map(lambda a,b: a+b, marker0, (X+markersize/2, -Y-markersize/2)))]
    marker4 = [tuple(map(lambda a,b: a+b, marker0, (-X,-Y))), 
               tuple(map(lambda a,b: a+b, marker0, (-X-markersize, -Y-markersize))), 
               tuple(map(lambda a,b: a+b, marker0, (-X-markersize/2, -Y-markersize/2)))]
    return (marker1, marker2, marker3, marker4)


class Config:
    # Layer configuration
    LAYER_NAMES = ['0', 'add', 'drop', 'res', 'ss1', 'ss2', 'taper', 'ind', 
                'mark', 'label', 'window', 'chip', 'pulley', 'turn', 'holes', 'grating']
    
    # Character conversion table
    CONV_TABLE = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                'A', 'B', 'C', 'D', 'E', 'F')
    P_TABLE = {'0', '3', '7', '8', 'C', 'F'}
    
    # Drawing tolerances
    ARC_TOL = 0.000001
    EULER_TOL = 0.000001
    
    # Device parameters
    DEV_STEP = 0.003
    DEV_NAME = 'resonator_array'
    
    # Chip dimensions
    CHIP_WIDTH = 12000.
    CHIP_HEIGHT = 12000.
    STREET_WIDTH = 40.
    
    # Window dimensions        
    window_x = 0.
    window_y = 0.
    WINDOW_CLEARANCE_X = 500.
    WINDOW_CLEARANCE_Y = 625. + 450/2.
    WINDOW_P_WIDTH = 8000.
    WINDOW_P_HEIGHT = 8000.


# def _create_grating_coupler(self, position, rotation_deg=0, mirror_x=False):
#     """Create a grating coupler at specified position with transformation."""
#     # Load the existing GDS file
#     lib = gd.GdsLibrary(infile="gratingcoupler.gds")
#     gc_cell = lib.cells["GC"]
#     offset_dist = [0,0]
#     position =  [a - b for a, b in zip(position, offset_dist)]
#     # Get all polygons from the grating coupler cell
#     polygons_by_spec = gc_cell.get_polygons(by_spec=True)
    
#     # Create a new cell to hold the transformed grating coupler
#     gc_holder = gd.Cell(f"GC_HOLDER_{position[0]}_{position[1]}")
    
#     # Process each polygon in the grating coupler
#     for (layer, datatype), poly_list in polygons_by_spec.items():
#         for polygon in poly_list:
#             # Apply transformation to each polygon
#             pts_trans = transform_points(polygon, origin=(0, 0), 
#                                        rotation_deg=rotation_deg, 
#                                        mirror_x=mirror_x)
#             # Translate to final position
#             pts_trans += position
#             # Add transformed polygon to holder cell
#             gc_holder.add(gd.Polygon(pts_trans, layer=layer, datatype=datatype))
    
#     # Add the holder cell to our library
#     self.cell_list.append(gc_holder)
#     self.cell_names.append(gc_holder.name)
    
#     # Add the holder cell to the grating layer
#     self.cell_list[self.cell_names.index('grating')].add(gc_holder)
    
#     return gc_holder


class EulerBend:
    """Helper class for Euler bend calculations."""
    @staticmethod
    def alpha(d, p):
        return (2 * np.sqrt(2) * integrate.quad(lambda t: np.sin(t**2), 0, np.sqrt((np.pi * p)/2))[0] + \
            2 / np.sqrt(np.pi * p) * np.sin((np.pi * (1 - p)) / 2))**2 / d**2
    
    @staticmethod
    def sp(d, p):
        return np.sqrt(np.pi * p / EulerBend.alpha(d, p))
    
    @staticmethod
    def rp(d, p):
        return 1 / np.sqrt(np.pi * p * EulerBend.alpha(d, p))
    
    @staticmethod
    def ys(d, p, s):
        return integrate.quad(lambda t: np.cos(EulerBend.alpha(d, p) * t**2 / 2), 0, s)[0]
    
    @staticmethod
    def xs(d, p, s):
        return integrate.quad(lambda t: np.sin(EulerBend.alpha(d, p) * t**2 / 2), 0, s)[0]


class DeviceParameters:
    """Container for device parameters with default values."""
    def __init__(self):
        # Device 0 parameters
        self.start_wg_W_0 = 1.0
        self.shift_wg_W_0 = 0.0
        self.shift_wg_W_k_0 = 0.0
        self.start_ad_wg_W_0 = 0.701
        self.target_rad_0 = 16.025
        self.offset_rad_0 = 0.
        self.start_rad_0 = self.target_rad_0 - self.offset_rad_0
        self.shift_rad_0 = 0.
        self.shift_rad_k_0 = 0.
        self.target_wg_L_0 = 0.3
        self.delta_wg_L_0 = 0.72
        self.offset_wg_L_0 = round(round(self.delta_wg_L_0/2, 3) // Config.DEV_STEP) * Config.DEV_STEP
        self.start_wg_L_0 = round((self.target_wg_L_0 - self.offset_wg_L_0) // Config.DEV_STEP) * Config.DEV_STEP
        self.shift_wg_L_0 = round(round(self.delta_wg_L_0/3, 3) // Config.DEV_STEP) * Config.DEV_STEP
        self.shift_wg_L_k_0 = round(round(self.delta_wg_L_0/3/16, 3) // Config.DEV_STEP) * Config.DEV_STEP
        self.start_gap_0 = 0.201
        self.shift_gap_0 = 0.0
        self.shift_gap_k_0 = 0.0
        self.start_clength_0 = 1.95
        self.shift_clength_0 = 0.0
        self.shift_clength_k_0 = 0.0
        self.start_hradh_0 = 5.0
        self.shift_hradh_0 = 0.0
        self.shift_hradh_k_0 = 0.0
        self.start_hradv_0 = 10.0
        self.shift_hradv_0 = 0.0
        self.shift_hradv_k_0 = 0.0
        self.bend_radius = 40
        self.bend_direction = 'down'
        self.final_length = 0

        # Array parameters
        self.array_ino = 0  # upright=0, invert=1
        self.array_flip = 1  # enable=1, disable=0
        self.shift_resonator_X = self.target_rad_0 * 60 / 16
        self.array_area_W = 100.
        self.start_resonator_clearance = Config.window_x #0.5 * (Config.WINDOW_P_WIDTH - self.array_area_W)
        self.array_resonator_X = ma.floor(self.array_area_W / self.shift_resonator_X)
        
        # Added parameters from resonator array code
        self.dev_H = 150
        self.dev_aH = 140
        self.start_dev_X = -self.array_area_W/2
        self.start_dev_Y = -0.5*Config.WINDOW_P_HEIGHT + 500
        self.shift_dev_Y = self.dev_H
        self.start_dev_aY = self.start_dev_Y
        self.shift_dev_aY = self.dev_aH
        
        # Waveguide parameters
        self.ad_wg_taper_buffer = 0.
        self.ad_wg_rampcle = 40.
        self.ad_wg_clearance = 10.
        self.ad_wg_bezcoe = 0.3
        self.start_ad_neck_L = 75.
        self.start_ad_waist_L = 0.
        self.start_ad_waist_W = 1.0
        self.start_ad_waist_L_top = 100.0
        self.start_ad_taper_W = 1.0
        self.ad_wg_start = -100    
        
        # Drawing parameters
        self.flin_pts = 800
        self.plin1_pts = 785
        self.plin2_pts = 30

#def Ring(chip,structure,params=DeviceParameters(),bgcolor=None,**kwargs):
class Ring(SolidPline):
    '''
    Ring Resonator with Euler band
    '''

    name="EULERRING"

    def __init__(self,insert, params=DeviceParameters(),parity=0, rotation=0.,color=const.BYLAYER,bgcolor=None,layer='0',linetype=None,ralign=const.BOTTOM,valign=const.BOTTOM,vflip=False,hflip=False, w=1, **kwargs):

        self.insert = insert
        self.rotation = ma.radians(rotation)
        self.color = color
        self.bgcolor = bgcolor
        self.layer = layer
        self.linetype = linetype
        self.width = w
        
        self.points = []
        self.valign = valign
        self.ralign = ralign
        self.parity = parity
        
        self.vflip = vflip and -1 or 1
        self.hflip = hflip and -1 or 1
        
        
        # Initialize counters
        self.text_ix = 0
        self.gov_ix = 0
        self.ar_ix = 0
        self.tap_ix = 0
        self.res_ix = 0
        self.res_ix_0 = 0
        

        self.params = params

        # Initialize device parameters
        self.devices = [0]
        self.dev_num = len(self.devices)    
        self.fc_num_tot = 30
        self.ar_row_tot = 16
        self.ar_row_h = self.ar_row_tot / 2
        self.fc_cur = 0
        
        # Initialize counters
        self.text_ix = 0
        self.gov_ix = 0
        self.ar_ix = 0
        self.tap_ix = 0
        self.res_ix = 0
        self.res_ix_0 = 0

        # Initialize positions
        dev_X = self.params.start_dev_X
        dev_Y = self.params.start_dev_Y + self.params.shift_dev_Y * self.gov_ix
        dev_aY = self.params.start_dev_aY + self.params.shift_dev_aY * self.ar_ix - dev_Y
        start_wg_W_ = self.params.start_ad_waist_W
        ad_wg_W = self.params.start_ad_wg_W_0
        
        # Array settings (exact from reference)
        sign_k_ini = self.params.array_ino  # upright=0, invert=1
        flip_array = self.params.array_flip  # enable=1, disable=04

    def _get_radius_align(self):

        #by default the radius is defined as the inner radius
        if self.ralign == const.MIDDLE:
            dr = -self.height/2.
        elif self.ralign == const.TOP:
            dr = -self.height
        else:  # const.BOTTOM
            dr = 0.

        return (0, dr)

    def _get_align_vector(self):

        #note: vertical alignment is flipped from regular rectangle
        if self.valign == const.MIDDLE:
            dy = -self.height/2.
        elif self.valign == const.TOP:
            dy = -self.height
        else:  # const.BOTTOM
            dy = 0.

        return (0, dy)

    def _build(self):
        data = DXFList()
        ralign = self._get_radius_align()
        self.points = self._calc_points(self.parity)
        
        align_vector = list(self._get_align_vector())
        offset_x = 0.5*(np.max(self.points[:,0]) - np.min(self.points[:,0]))
        offset_y = 0.5*(np.max(self.points[:,1]) + np.min(self.points[:,1]))
        print(align_vector)
        print(f"offset x ={offset_x}")
        print(f"offset y = {offset_y}")
        # update align vector with ring geometry offset
        align_vector[0] -= offset_x
        align_vector[1] -= offset_y
        
        self._transform_points(align_vector)
        if self.color is not None:
            data.append(self._build_polyline())
        if self.bgcolor is not None:
            #if _calc_points has been run, rmin is already set
            if self.rmin <= 0:
                #if self.angle%(2*math.pi) == math.radians(90): #rounded corner case
                for i in range(self.segments+1):
                    data.append(self._build_solid_triangle(i))
            else: #rmin>0, normal operation
                for i in range(self.segments+1):
                    data.append(self._build_solid_quad(i))
        return data
        
    def _transform_points(self,align):
        self.points = [vadd(self.insert,  # move to insert point
                            rotate_2d(  # rotate at origin
                                ((point[0]+align[0])*self.hflip,(point[1]+align[1])*self.vflip), self.rotation))
                       for point in self.points]

    def _calc_points(self, parity):
        k = parity
        #dev_Y_shift = k // self.params.array_resonator_X * self.params.shift_dev_Y
        #dev_aY_shift = k // self.params.array_resonator_X * self.params.shift_dev_aY
        
        # Calculate resonator parameters (exact from reference)
        res_W = (self.params.start_wg_W_0 + self.params.shift_wg_W_0 * self.res_ix_0 + self.params.shift_wg_W_k_0 * k)
        
        res_length = (self.params.start_wg_L_0 + self.params.shift_wg_L_0 * self.res_ix_0 + self.params.shift_wg_L_k_0 * k)
        
        rad = (self.params.start_rad_0 + self.params.shift_rad_0 * self.res_ix_0 + self.params.shift_rad_k_0 * k)
        
        hradh = (self.params.start_hradh_0 + self.params.shift_hradh_0 * self.res_ix_0 + self.params.shift_hradh_k_0 * k)
        
        hradv = (self.params.start_hradv_0 + self.params.shift_hradv_0 * self.res_ix_0 + self.params.shift_hradv_k_0 * k)
        
        gap = (self.params.start_gap_0 + self.params.shift_gap_0 * self.res_ix_0 + self.params.shift_gap_k_0 * k)
        
        clength = (self.params.start_clength_0 + self.params.shift_clength_0 * self.res_ix_0 + self.params.shift_clength_k_0 * k)
        
        # Pulley calculations (exact from reference)
        pulley_rad = rad + 0.5 * (gap + res_W)
        #pulley_wg_rad = rad + gap + 0.5 * (ad_wg_W + res_W)
        pulley_theda = clength / pulley_rad
        #pulley_brad = (0.5 * self.params.shift_resonator_X - pulley_wg_rad * np.sin(0.5 * pulley_theda)) / np.sin(0.5 * pulley_theda)
        #pulley_shift_Y = (pulley_wg_rad + pulley_brad) * (1 - np.cos(0.5 * pulley_theda))
        
        # Create resonator path (exact from reference)
        #res_spec = set_layer_spec(self.config.LAYER_NAMES, self.spec, 'res')
        dia = rad * 2.
        ep = 1 - pulley_theda / np.pi
        
        flin = np.linspace(0, EulerBend.sp(dia, 1), self.params.flin_pts)
        fex = np.array([EulerBend.xs(dia, 1, i) for i in flin])
        fey = np.array([EulerBend.ys(dia, 1, i) for i in flin])
        
        plin1 = np.linspace(0, EulerBend.sp(dia, ep), self.params.plin1_pts)
        plin2 = np.linspace(np.pi/2 * ep, np.pi/2, self.params.plin2_pts)
        
        
        pex1 = np.array([EulerBend.xs(dia, ep, i) for i in plin1])
        pey1 = np.array([EulerBend.ys(dia, ep, i) for i in plin1])
        pex2 = np.array([dia/2 - EulerBend.rp(dia, ep) * np.sin(np.pi/2 - i) for i in plin2])
        pey2 = np.array([EulerBend.ys(dia, ep, EulerBend.sp(dia, ep)) - 
                    EulerBend.rp(dia, ep) * np.cos(np.pi/2 * (1 - ep)) + 
                    EulerBend.rp(dia, ep) * np.cos(np.pi/2 - i) for i in plin2])
        
        # self.offsetex = np.max(pex2)
        # self.offsetey = np.max(pey2)
        
        pexline = np.concatenate(([fex[0]], 
                                  fex, 
                                  dia - np.flip(fex)[1:], 
                                  dia - pex1[1:], 
                                  dia - pex2[1:], 
                                  np.flip(pex2)[1:], 
                                  np.flip(pex1)[1:]))
        
        peyline = np.concatenate(([fey[0]], 
                                   fey + res_length, 
                                   np.flip(fey)[1:] + res_length, 
                                  -pey1[1:], 
                                  -pey2[1:], 
                                   np.flip(-pey2)[1:], 
                                   np.flip(-pey1)[1:]))
        
        peline = np.column_stack((pexline, peyline))
        return peline

    def _build_polyline(self):
        '''Build the polyline (key component)'''
        polyline = Polyline(self.points, startwidth=self.width, endwidth=self.width ,layer=self.layer,color=self.color, flags=0)
        #polyline.close() #redundant in most cases
        if self.linetype is not None:
            polyline['linetype'] = self.linetype
        return polyline
    
    def _build_solid_quad(self,i,center=None):
        ''' build a single background solid quadrangle segment '''
        solidpts = [self.points[j] for j in [i,i+1,-i-2,-i-1]]
        return Solid(solidpts, color=self.bgcolor, layer=self.layer)  

    
    def _build_solid_triangle(self,i):
        ''' build a single background solid quadrangle segment '''
        solidpts = [self.points[j] for j in [0,i+1,i+2]]
        return Solid(solidpts, color=self.bgcolor, layer=self.layer)  
    
    def __dxf__(self):
        ''' get the dxf string '''
        return dxfstr(self.__dxftags__())
    
    def __dxftags__(self):
        return self._build() 


# ===============================================================================
# basic POSITIVE microstrip function definitions
# ===============================================================================

def Strip_straight(chip,structure,length,w=None,bgcolor=None,**kwargs): #note: uses CPW conventions
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if bgcolor is None:
        bgcolor = chip.wafer.bg()
    if w is None:
        try:
            w = struct().defaults['w']
        except KeyError:
            print('\x1b[33ms not defined in ',chip.chipID,'!\x1b[0m')
    
    chip.add(dxf.rectangle(struct().start,length,w,valign=const.MIDDLE,rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)),structure=structure,length=length)

def Strip_taper(chip,structure,length=None,w0=None,w1=None,bgcolor=None,offset=(0,0),**kwargs): #note: uses CPW conventions
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if bgcolor is None:
        bgcolor = chip.wafer.bg()
    if w0 is None:
        try:
            w0 = struct().defaults['w']
        except KeyError:
            print('\x1b[33ms not defined in ',chip.chipID,'!\x1b[0m')
    if w1 is None:
        try:
            w1 = struct().defaults['w']
        except KeyError:
            print('\x1b[33ms not defined in ',chip.chipID,'!\x1b[0m')
    #if undefined, make outer angle 30 degrees
    if length is None:
        length = math.sqrt(3)*abs(w0/2-w1/2)
    
    chip.add(SkewRect(struct().start,length,w0,offset,w1,rotation=struct().direction,valign=const.MIDDLE,edgeAlign=const.MIDDLE,bgcolor=bgcolor,**kwargs),structure=structure,offsetVector=vadd((length,0),offset))

def Strip_bend(chip,structure,angle=90,CCW=True,w=None,radius=None,ptDensity=120,bgcolor=None,**kwargs):
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if w is None:
        try:
            w = struct().defaults['w']
        except KeyError:
            print('\x1b[33mw not defined in ',chip.chipID)
    if radius is None:
        try:
            radius = struct().defaults['radius']
        except KeyError:
            print('\x1b[33mradius not defined in ',chip.chipID,'!\x1b[0m')
            return
    if bgcolor is None:
        bgcolor = chip.wafer.bg()
        
    while angle < 0:
        angle = angle + 360
    angle = angle%360
        
    chip.add(CurveRect(struct().start,w,radius,angle=angle,ptDensity=ptDensity,ralign=const.MIDDLE,valign=const.MIDDLE,rotation=struct().direction,vflip=not CCW,bgcolor=bgcolor,**kwargs))
    struct().updatePos(newStart=struct().getPos((radius*math.sin(math.radians(angle)),(CCW and 1 or -1)*radius*(math.cos(math.radians(angle))-1))),angle=CCW and -angle or angle)


def Strip_stub_open(chip,structure,flipped=False,curve_out=True,r_out=None,w=None,allow_oversize=True,length=None,bgcolor=None,**kwargs):
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if w is None:
        try:
            w = struct().defaults['w']
        except KeyError:
            print('\x1b[33ms not defined in ',chip.chipID,'!\x1b[0m')
    if r_out is None:
        try:
            if allow_oversize:
                r_out = struct().defaults['r_out']
            else:
                r_out = min(struct().defaults['r_out'],w/2)
        except KeyError:
            print('r_out not defined in ',chip.chipID,'!\x1b[0m')
            r_out=0
    if bgcolor is None:
        bgcolor = chip.wafer.bg()
    
    
    if r_out > 0:
        dx = 0.
        if flipped:
            if allow_oversize:
                dx = r_out
            else:
                dx = min(w/2,r_out)
        
        if allow_oversize:
            l=r_out
        else:
            l=min(w/2,r_out)
        
        if length is None: length=0

        chip.add(RoundRect(struct().getPos((dx,0)),max(length,l),w,l,roundCorners=[0,curve_out,curve_out,0],hflip=flipped,valign=const.MIDDLE,rotation=struct().direction,bgcolor=bgcolor,**kwargs),structure=structure,length=l)
    else:
        if length is not None:
            if allow_oversize:
                l=length
            else:
                l=min(w/2,length)
        else:
            l=w/2
        Strip_straight(chip,structure,l,w=w,bgcolor=bgcolor,**kwargs)

def Strip_stub_short(chip,structure,r_ins=None,w=None,flipped=False,extra_straight_section=False,bgcolor=None,**kwargs):
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if w is None:
        try:
            w = struct().defaults['w']
        except KeyError:
            print('\x1b[33mw not defined in ',chip.chipID,'!\x1b[0m')
    if r_ins is None:
        try:
            r_ins = struct().defaults['r_ins']
        except KeyError:
            #print('r_ins not defined in ',chip.chipID,'!\x1b[0m')
            r_ins=0
    if bgcolor is None:
        bgcolor = chip.wafer.bg()
    
    if r_ins > 0:
        if extra_straight_section and not flipped:
            Strip_straight(chip, struct(), r_ins, w=w,rotation=struct().direction,bgcolor=bgcolor,**kwargs)
        chip.add(InsideCurve(struct().getPos((0,-w/2)),r_ins,rotation=struct().direction,hflip=flipped,bgcolor=bgcolor,**kwargs))
        chip.add(InsideCurve(struct().getPos((0,w/2)),r_ins,rotation=struct().direction,hflip=flipped,vflip=True,bgcolor=bgcolor,**kwargs))
        if extra_straight_section and flipped:
                Strip_straight(chip, struct(), r_ins, w=w,rotation=struct().direction,bgcolor=bgcolor,**kwargs)

def Strip_pad(chip,structure,length,r_out=None,w=None,bgcolor=None,**kwargs):
    '''
    Draw a pad with all rounded corners (similar to strip_stub_open + strip_straight + strip_stub_open but only one shape)

    '''
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if w is None:
        try:
            w = struct().defaults['w']
        except KeyError:
            print('\x1b[33ms not defined in ',chip.chipID,'!\x1b[0m')
    if r_out is None:
        try:
            r_out = min(struct().defaults['r_out'],w/2)
        except KeyError:
            print('r_out not defined in ',chip.chipID,'!\x1b[0m')
            r_out=0
    if bgcolor is None:
        bgcolor = chip.wafer.bg()
    
    
    if r_out > 0:
        chip.add(RoundRect(struct().getPos((0,0)),length,w,r_out,roundCorners=[1,1,1,1],valign=const.MIDDLE,rotation=struct().direction,bgcolor=bgcolor,**kwargs),structure=structure,length=length)
    else:
        Strip_straight(chip,structure,length,w=w,bgcolor=bgcolor,**kwargs)


# ===============================================================================
#  NEGATIVE wire/stripline function definitions
# ===============================================================================

def Wire_bend(chip,structure,angle=90,CCW=True,w=None,radius=None,bgcolor=None,**kwargs):
    #only defined for 90 degree bends
    if angle%90 != 0:
        return
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if w is None:
        try:
            w = struct().defaults['w']
        except KeyError:
            print('\x1b[33mw not defined in ',chip.chipID)
    if radius is None:
        try:
            radius = struct().defaults['radius']
        except KeyError:
            print('\x1b[33mradius not defined in ',chip.chipID,'!\x1b[0m')
            return
    if bgcolor is None:
        bgcolor = chip.wafer.bg()
        
    while angle < 0:
        angle = angle + 360
    angle = angle%360
        
    if radius-w/2 > 0:
        chip.add(CurveRect(struct().start,radius-w/2,radius,angle=angle,roffset=-w/2,ralign=const.TOP,valign=const.TOP,rotation=struct().direction,vflip=not CCW,bgcolor=bgcolor,**kwargs))
    for i in range(angle//90):
        chip.add(InsideCurve(struct().getPos(vadd(rotate_2d((radius+w/2,(CCW and 1 or -1)*(radius+w/2)),(CCW and -1 or 1)*math.radians(i*90)),(0,CCW and -radius or radius))),radius+w/2,rotation=struct().direction+(CCW and -1 or 1)*i*90,bgcolor=bgcolor,vflip=not CCW,**kwargs))
    struct().updatePos(newStart=struct().getPos((radius*math.sin(math.radians(angle)),(CCW and 1 or -1)*radius*(math.cos(math.radians(angle))-1))),angle=CCW and -angle or angle)


# ===============================================================================
# Grating coupler launcher
# ===============================================================================
def GCoupler_launcher():
    return 0

# ===============================================================================
# composite CPW function definitions
# ===============================================================================
def CPW_pad(chip,struct,l_pad=0,l_gap=0,padw=300,pads=50,l_lead=None,w=None,s=None,r_ins=None,r_out=None,bgcolor=None,**kwargs):
    if w is None:
        try:
            w = struct().defaults['w']
        except KeyError:
            w=0
            print('\x1b[33mw not defined in ',chip.chipID)
    CPW_stub_open(chip,struct,length=max(l_gap,pads),r_out=r_out,r_ins=r_ins,w=padw,s=pads,flipped=True,**kwargs)
    CPW_straight(chip,struct,max(l_pad,padw),w=padw,s=pads,**kwargs)
    if l_lead is None:
        l_lead = max(l_gap,pads)
    CPW_stub_short(chip,struct,length=l_lead,r_out=r_out,r_ins=r_ins,w=w,s=pads+padw/2-w/2,flipped=False,curve_ins=False,**kwargs)


def CPW_launcher(chip,struct,l_taper=None,l_pad=0,l_gap=0,padw=300,pads=160,w=None,s=None,r_ins=0,r_out=0,bgcolor=None,**kwargs):
    CPW_stub_open(chip,struct,length=max(l_gap,pads),r_out=r_out,r_ins=r_ins,w=padw,s=pads,flipped=True,**kwargs)
    CPW_straight(chip,struct,max(l_pad,padw),w=padw,s=pads,**kwargs)
    CPW_taper(chip,struct,length=l_taper,w0=padw,s0=pads,**kwargs)
    


def CPW_taper_cap(chip,structure,gap,width,l_straight=0,l_taper=None,s1=None,**kwargs):
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if s1 is None:
        try:
            s = struct().defaults['s']
            w = struct().defaults['w']
            s1 = width*s/w
        except KeyError:
            print('\x1b[33mw not defined in ',chip.chipID)
            print('\x1b[33ms not defined in ',chip.chipID)
    if l_taper is None:
        l_taper = 3*width
    if l_straight<=0:
        try:
            tap_angle = math.degrees(math.atan(2*l_taper/(width-struct().defaults['w'])))
        except KeyError:
            print('\x1b[33mw not defined in ',chip.chipID)
            tap_angle = 90
    else:
        tap_angle = 90
        
    CPW_taper(chip,structure,length=l_taper,w1=width,s1=s1,**kwargs)
    if l_straight > 0 :
        CPW_straight(chip,structure,length=l_straight,w=width,s=s1,**kwargs)
    CPW_cap(chip, structure, gap, w=width, s=s1, angle=tap_angle, **kwargs)
    if l_straight > 0 :
        CPW_straight(chip,structure,length=l_straight,w=width,s=s1,**kwargs)
    CPW_taper(chip,structure,length=l_taper,w0=width,s0=s1,**kwargs)
    
def CPW_directTo(chip,from_structure,to_structure,to_flipped=True,w=None,s=None,radius=None,CW1_override=None,CW2_override=None,flip_angle=False,debug=False,**kwargs):
    def struct1():
        if isinstance(from_structure,m.Structure):
            return from_structure
        else:
            return chip.structure(from_structure)
    if radius is None:
        try:
            radius = struct1().defaults['radius']
        except KeyError:
            print('\x1b[33mradius not defined in ',chip.chipID,'!\x1b[0m')
            return
    #struct2 is only a local copy
    struct2 = isinstance(to_structure,m.Structure) and to_structure or chip.structure(to_structure)
    if to_flipped:
        struct2.direction=(struct2.direction+180.)%360
    
    CW1 = vector2angle(struct1().getGlobalPos(struct2.start)) < 0
    CW2 = vector2angle(struct2.getGlobalPos(struct1().start)) < 0

    target1 = struct1().getPos((0,CW1 and -2*radius or 2*radius))
    target2 = struct2.getPos((0,CW2 and -2*radius or 2*radius))
    
    #reevaluate based on center positions
    
    CW1 = vector2angle(struct1().getGlobalPos(target2)) < 0
    CW2 = vector2angle(struct2.getGlobalPos(target1)) < 0
    
    if CW1_override is not None:
        CW1 = CW1_override
    if CW2_override is not None:
        CW2 = CW2_override

    center1 = struct1().getPos((0,CW1 and -radius or radius))
    center2 = struct2.getPos((0,CW2 and -radius or radius))
    
    if debug:
        chip.add(dxf.line(struct1().getPos((-3000,0)),struct1().getPos((3000,0)),layer='FRAME'))
        chip.add(dxf.line(struct2.getPos((-3000,0)),struct2.getPos((3000,0)),layer='FRAME'))
        chip.add(dxf.circle(center=center1,radius=radius,layer='FRAME'))
        chip.add(dxf.circle(center=center2,radius=radius,layer='FRAME'))
    
    correction_angle=math.asin(abs(2*radius*(CW1 - CW2)/distance(center2,center1)))
    angle1 = abs(struct1().direction - math.degrees(vector2angle(vsub(center2,center1)))) + math.degrees(correction_angle)
    if flip_angle:
        angle1 = 360-abs(struct1().direction - math.degrees(vector2angle(vsub(center2,center1)))) + math.degrees(correction_angle)
    
    if debug:
        print(CW1,CW2,angle1,math.degrees(correction_angle))
    
    if angle1 > 270:
        if debug:
            print('adjusting to shorter angle')
        angle1 = min(angle1,abs(360-angle1))
    '''
    if CW1 - CW2 == 0 and abs(angle1)>100:
        if abs((struct1().getGlobalPos(struct2.start))[1]) < 2*radius:
            print('adjusting angle')
            angle1 = angle1 + math.degrees(math.asin(abs(2*radius/distance(center2,center1))))
            '''
    CPW_bend(chip,from_structure,angle=angle1,w=w,s=s,radius=radius, CCW=CW1,**kwargs)
    CPW_straight(chip,from_structure,distance(center2,center1)*math.cos(correction_angle),w=w,s=s,**kwargs)
    
    angle2 = abs(struct1().direction-struct2.direction)
    if angle2 > 270:
        angle2 = min(angle2,abs(360-angle2))
    CPW_bend(chip,from_structure,angle=angle2,w=w,s=s,radius=radius,CCW=CW2,**kwargs)

#Various wiggles (meander) definitions 

def wiggle_calc(chip,structure,length=None,nTurns=None,maxWidth=None,Width=None,start_bend = True,stop_bend=True,w=None,s=None,radius=None,debug=False,**kwargs):
    #figure out 
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if radius is None:
        try:
            radius = struct().defaults['radius']
        except KeyError:
            print('\x1b[33mradius not defined in ',chip.chipID,'!\x1b[0m')
            return
    if w is None:
        try:
            w = struct().defaults['w']
        except KeyError:
            print('\x1b[33mw not defined in ',chip.chipID,'!\x1b[0m')
    if s is None:
        try:
            s = struct().defaults['s']
        except KeyError:
            s = 0
    #prevent dumb entries
    if nTurns is None:
        nTurns = 1
    elif nTurns < 1:
        nTurns = 1
    
    #debug
    if debug:
        print('w=',w,' s=',s,' nTurns=',nTurns,' length=',length,' Width=',Width,' maxWidth=',maxWidth)
    
    #is length constrained?
    if length is not None:
        if nTurns is None:
            nTurns = 1
        h = (length - nTurns*2*math.pi*radius - (start_bend+stop_bend)*(math.pi/2-1)*radius)/(4*nTurns)

        #is width constrained?
        if Width is not None or maxWidth is not None:
            #maxWidth corresponds to the wiggle width, while Width corresponds to the total width filled
            if maxWidth is not None:
                if Width is None:
                    Width = maxWidth
                else:
                    maxWidth = min(maxWidth,Width)
            else:
                maxWidth = Width
            while h+radius+w/2+s/2>maxWidth:
                nTurns = nTurns+1
                h = (length - nTurns*2*math.pi*radius - (start_bend+stop_bend)*(math.pi/2-1)*radius)/(4*nTurns)
    else: #length is not contrained
        h= maxWidth-radius-w/2-s
    h = max(h,radius)
    return {'nTurns':nTurns,'h':h,'length':length,'maxWidth':maxWidth,'Width':Width}

def CPW_wiggles(chip,structure,length=None,nTurns=None,maxWidth=None,CCW=True,start_bend = True,stop_bend=True,w=None,s=None,radius=None,bgcolor=None,debug=False,**kwargs):
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if radius is None:
        try:
            radius = struct().defaults['radius']
        except KeyError:
            print('\x1b[33mradius not defined in ',chip.chipID,'!\x1b[0m')
            return
    
    params = wiggle_calc(chip,structure,length,nTurns,maxWidth,None,start_bend,stop_bend,w,s,radius,**kwargs)
    [nTurns,h,length,maxWidth]=[params[key] for key in ['nTurns','h','length','maxWidth']]
    if (length is None) or (h is None) or (nTurns is None):
        print('not enough params specified for CPW_wiggles!')
        return
    if debug:
        chip.add(dxf.rectangle(struct().start,(nTurns*4 + start_bend + stop_bend)*radius,2*h,valign=const.MIDDLE,rotation=struct().direction,layer='FRAME'))
        chip.add(dxf.rectangle(struct().start,(nTurns*4 + start_bend + stop_bend)*radius,2*maxWidth,valign=const.MIDDLE,rotation=struct().direction,layer='FRAME'))
    if start_bend:
        CPW_bend(chip,structure,angle=90,CCW=CCW,w=w,s=s,radius=radius,bgcolor=bgcolor,**kwargs)
        if h > radius:
            CPW_straight(chip,structure,h-radius,w=w,s=s,bgcolor=bgcolor,**kwargs)
    else:
        CPW_straight(chip,structure,h,w=w,s=s,bgcolor=bgcolor,**kwargs)
    CPW_bend(chip,structure,angle=180,CCW=not CCW,w=w,s=s,radius=radius,bgcolor=bgcolor,**kwargs)
    CPW_straight(chip,structure,h+radius,w=w,s=s,bgcolor=bgcolor,**kwargs)
    if h > radius:
        CPW_straight(chip,structure,h-radius,w=w,s=s,bgcolor=bgcolor,**kwargs)
    CPW_bend(chip,structure,angle=180,CCW=CCW,w=w,s=s,radius=radius,bgcolor=bgcolor,**kwargs)
    if h > radius:
        CPW_straight(chip,structure,h-radius,w=w,s=s,bgcolor=bgcolor,**kwargs)
    for n in range(nTurns-1):
        CPW_straight(chip,structure,h+radius,w=w,s=s,bgcolor=bgcolor,**kwargs)
        CPW_bend(chip,structure,angle=180,CCW=not CCW,w=w,s=s,radius=radius,bgcolor=bgcolor,**kwargs)
        CPW_straight(chip,structure,h+radius,w=w,s=s,bgcolor=bgcolor,**kwargs)
        if h > radius:
            CPW_straight(chip,structure,h-radius,w=w,s=s,bgcolor=bgcolor,**kwargs)
        CPW_bend(chip,structure,angle=180,CCW=CCW,w=w,s=s,radius=radius,bgcolor=bgcolor,**kwargs)
        if h > radius:
            CPW_straight(chip,structure,h-radius,w=w,s=s,bgcolor=bgcolor,**kwargs)
    if stop_bend:
        CPW_bend(chip,structure,angle=90,CCW=not CCW,w=w,s=s,radius=radius,bgcolor=bgcolor,**kwargs)
    else:
        CPW_straight(chip,structure,radius,w=w,s=s,bgcolor=bgcolor,**kwargs)

def Strip_wiggles(chip,structure,length=None,nTurns=None,maxWidth=None,CCW=True,start_bend = True,stop_bend=True,w=None,radius=None,bgcolor=None,**kwargs):
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if radius is None:
        try:
            radius = struct().defaults['radius']
        except KeyError:
            print('\x1b[33mradius not defined in ',chip.chipID,'!\x1b[0m')
            return
    
    params = wiggle_calc(chip,structure,length,nTurns,maxWidth,None,start_bend,stop_bend,w,0,radius,**kwargs)
    [nTurns,h,length,maxWidth]=[params[key] for key in ['nTurns','h','length','maxWidth']]
    if (h is None) or (nTurns is None):
        print('not enough params specified for Microstrip_wiggles!')
        return

    if start_bend:
        Strip_bend(chip,structure,angle=90,CCW=CCW,w=w,radius=radius,bgcolor=bgcolor,**kwargs)
        if h > radius:
            Strip_straight(chip,structure,h-radius,w=w,bgcolor=bgcolor,**kwargs)
    else:
        Strip_straight(chip,structure,h,w=w,bgcolor=bgcolor,**kwargs)
    Strip_bend(chip,structure,angle=180,CCW=not CCW,w=w,radius=radius,bgcolor=bgcolor,**kwargs)
    Strip_straight(chip,structure,h+radius,w=w,bgcolor=bgcolor,**kwargs)
    if h > radius:
        Strip_straight(chip,structure,h-radius,w=w,bgcolor=bgcolor,**kwargs)
    Strip_bend(chip,structure,angle=180,CCW=CCW,w=w,radius=radius,bgcolor=bgcolor,**kwargs)
    if h > radius:
        Strip_straight(chip,structure,h-radius,w=w,bgcolor=bgcolor,**kwargs)
    for n in range(nTurns-1):
        Strip_straight(chip,structure,h+radius,w=w,bgcolor=bgcolor,**kwargs)
        Strip_bend(chip,structure,angle=180,CCW=not CCW,w=w,radius=radius,bgcolor=bgcolor,**kwargs)
        Strip_straight(chip,structure,h+radius,w=w,bgcolor=bgcolor,**kwargs)
        if h > radius:
            Strip_straight(chip,structure,h-radius,w=w,bgcolor=bgcolor,**kwargs)
        Strip_bend(chip,structure,angle=180,CCW=CCW,w=w,radius=radius,bgcolor=bgcolor,**kwargs)
        if h > radius:
            Strip_straight(chip,structure,h-radius,w=w,bgcolor=bgcolor,**kwargs)
    if stop_bend:
        Strip_bend(chip,structure,angle=90,CCW=not CCW,w=w,radius=radius,bgcolor=bgcolor,**kwargs)
    else:
        Strip_straight(chip,structure,radius,w=w,bgcolor=bgcolor,**kwargs)

def Inductor_wiggles(chip,structure,length=None,nTurns=None,maxWidth=None,Width=None,CCW=True,start_bend = True,stop_bend=True,pad_to_width=None,w=None,s=None,radius=None,bgcolor=None,**kwargs):
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if radius is None:
        try:
            radius = struct().defaults['radius']
        except KeyError:
            print('\x1b[33mradius not defined in ',chip.chipID,'!\x1b[0m')
            return
    if bgcolor is None:
        bgcolor = chip.wafer.bg()
    if w is None:
        try:
            w = struct().defaults['w']
        except KeyError:
            print('\x1b[33mw not defined in ',chip.chipID,'!\x1b[0m')
    if s is None:
        try:
            s = struct().defaults['s']
        except KeyError:
            print('\x1b[33ms not defined in ',chip.chipID,'!\x1b[0m')

    if pad_to_width is None and Width is not None:
        pad_to_width = True
    params = wiggle_calc(chip,structure,length,nTurns,maxWidth,Width,start_bend,stop_bend,w,0,radius,**kwargs)
    [nTurns,h,length,maxWidth,Width]=[params[key] for key in ['nTurns','h','length','maxWidth','Width']]
    if (h is None) or (nTurns is None):
        print('not enough params specified for CPW_wiggles!')
        return
    
    pm = (CCW - 0.5)*2
    
    #put rectangles on either side to line up with max width
    if pad_to_width:
        if Width is None:
            print('\x1b[33mERROR:\x1b[0m cannot pad to width with Width undefined!')
            return
        if start_bend:
            chip.add(dxf.rectangle(struct().getPos((0,h+radius+w/2)),(2*radius)+(nTurns)*4*radius,Width-(h+radius+w/2),rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)))
            chip.add(dxf.rectangle(struct().getPos((0,-h-radius-w/2)),(stop_bend)*(radius+w/2)+(nTurns)*4*radius + radius-w/2,(h+radius+w/2)-Width,rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)))
        else:
            chip.add(dxf.rectangle(struct().getPos((-h-radius-w/2,w/2)),(h+radius+w/2)-Width,(radius-w/2)+(nTurns)*4*radius,rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)))
            chip.add(dxf.rectangle(struct().getPos((h+radius+w/2,-radius)),Width-(h+radius+w/2),(stop_bend)*(radius+w/2)+(nTurns)*4*radius + w/2,rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)))
    #begin wiggles
    if start_bend:
        chip.add(dxf.rectangle(struct().getPos((0,pm*w/2)),radius+w/2,pm*(h+radius),rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)))
        Wire_bend(chip,structure,angle=90,CCW=CCW,w=w,radius=radius,bgcolor=bgcolor,**kwargs)
        if h > radius:
            chip.add(dxf.rectangle(struct().getPos((0,-pm*w/2)),h+w/2,pm*(radius-w/2),valign=const.BOTTOM,rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)),structure=struct(),length=h-radius)
        else:
            chip.add(dxf.rectangle(struct().getPos((0,-pm*w/2)),radius+w/2,pm*(radius-w/2),valign=const.BOTTOM,rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)))
    else:
        chip.add(dxf.rectangle(struct().getPos((0,-pm*w/2)),2*radius+w/2,pm*(radius-w/2),valign=const.BOTTOM,rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)),structure=struct(),length=radius)
        #struct().shiftPos(h)
    chip.add(dxf.rectangle(struct().getPos((0,pm*w/2)),-h-max(h,radius)-radius-w/2,pm*(2*radius-w),rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)))
    Wire_bend(chip,structure,angle=180,CCW=not CCW,w=w,radius=radius,bgcolor=bgcolor,**kwargs)
    struct().shiftPos(h+radius)
    if h > radius:
        struct().shiftPos(h-radius)
    chip.add(dxf.rectangle(struct().getPos((0,-pm*w/2)),-h-max(h,radius)-radius-w/2,pm*(-2*radius+w),rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)))
    Wire_bend(chip,structure,angle=180,CCW=CCW,w=w,radius=radius,bgcolor=bgcolor,**kwargs)
    if h > radius:
        struct().shiftPos(h-radius)
    for n in range(nTurns-1):
        struct().shiftPos(h+radius)
        chip.add(dxf.rectangle(struct().getPos((0,pm*w/2)),-h-max(h,radius)-radius-w/2,pm*(2*radius-w),rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)))
        Wire_bend(chip,structure,angle=180,CCW=not CCW,w=w,radius=radius,bgcolor=bgcolor,**kwargs)
        struct().shiftPos(2*h)
        chip.add(dxf.rectangle(struct().getPos((0,-pm*w/2)),-h-max(h,radius)-radius-w/2,pm*(-2*radius+w),rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)))
        Wire_bend(chip,structure,angle=180,CCW=CCW,w=w,radius=radius,bgcolor=bgcolor,**kwargs)
        struct().shiftPos(h-radius)
    chip.add(dxf.rectangle(struct().getLastPos((-radius-w/2,pm*w/2)),w/2+h,pm*(radius-w/2),rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)),structure=struct())
    if stop_bend:
        chip.add(dxf.rectangle(struct().getPos((radius+w/2,-pm*w/2)),h+radius,pm*(radius+w/2),rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)))
        Wire_bend(chip,structure,angle=90,CCW=not CCW,w=w,radius=radius,bgcolor=bgcolor,**kwargs)
    else:
        #CPW_straight(chip,structure,radius,w=w,s=s,bgcolor=bgcolor)
        chip.add(dxf.rectangle(struct().getPos((0,pm*w/2)),radius,pm*(radius-w/2),rotation=struct().direction,bgcolor=bgcolor,**kwargStrip(kwargs)),structure=struct(),length=radius)
        
def TwoPinCPW_wiggles(chip,structure,w=None,s_ins=None,s_out=None,s=None,Width=None,maxWidth=None,**kwargs):
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if w is None:
        try:
            w = struct().defaults['w']
        except KeyError:
            print('\x1b[33ms not defined in ',chip.chipID,'!\x1b[0m')
    if s is not None:
        #s overridden somewhere above
        if s_ins is None:
            s_ins = s
        if s_out is None:
            s_out = s
    if s_ins is None:
        try:
            s_ins = struct().defaults['s']
        except KeyError:
            print('\x1b[33ms not defined in ',chip.chipID,'!\x1b[0m')
    if s_out is None:
        if Width is not None:
            s_out = Width - w - s_ins/2
        else:
            try:
                s_out = struct().defaults['s']
            except KeyError:
                print('\x1b[33ms not defined in ',chip.chipID,'!\x1b[0m')
                
    s0 = struct().clone()
    maxWidth = wiggle_calc(chip,struct(),Width=Width,maxWidth=maxWidth,w=s_ins+2*w,s=0,**kwargs)['maxWidth']
    Inductor_wiggles(chip, s0, w=s_ins+2*w,Width=Width,maxWidth=maxWidth,**kwargs)
    Strip_wiggles(chip, struct(), w=s_ins,maxWidth=maxWidth-w,**kwargs)

def CPW_pincer(chip,structure,pincer_w,pincer_l,pincer_padw,pincer_tee_r=0,pad_r=None,w=None,s=None,pincer_flipped=False,bgcolor=None,**kwargs):
    '''
    pincer_w :      
    pincer_l :      length of pincer arms
    pincer_padw :   pincer pad trace width
    pincer_tee_r :  radius of tee
    pad_r:          inside radius of pincer bends
    w:              original trace width
    s:              pincer trace gap
    pincer_flipped: equivalent to hflip
    '''
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if w is None:
        try:
            w = struct().defaults['w']
        except KeyError:
            w=0
            print('\x1b[33mw not defined in ',chip.chipID)
    if s is None:
        try:
            s = struct().defaults['s']
        except KeyError:
            print('\x1b[33ms not defined in ',chip.chipID,'!\x1b[0m')
    if pad_r is None:
        pincer_r = pincer_padw/2 + s
        pad_r=0
    else:
        pincer_r = pincer_padw/2+s+abs(pad_r)#prevent negative entries
        pad_r = abs(pad_r)
        
    if not pincer_flipped: s_start = struct().clone()
    else:
        struct().shiftPos(pincer_padw+pincer_tee_r+2*s,angle=180)
        #struct().direction += 180
        s_start = struct().clone()

    s_left, s_right = CPW_tee(chip, struct(), w=w, s=s, w1=pincer_padw, s1=s, radius=pincer_tee_r + s, **kwargs)

    CPW_straight(chip, s_left, length=(pincer_w-w-2*s-2*pincer_tee_r)/2-pad_r, **kwargs)
    CPW_straight(chip, s_right, length=(pincer_w-w-2*s-2*pincer_tee_r)/2-pad_r, **kwargs)

    if pincer_l > s:
        CPW_bend(chip, s_left, CCW=True, w=pincer_padw, s=s, radius=pincer_r, **kwargs)
        CPW_straight(chip, s_left, length=pincer_l - s-pad_r, **kwargs)
        CPW_stub_open(chip, s_left, w=pincer_padw, s=s, **kwargs)

        CPW_bend(chip, s_right, CCW=False, w=pincer_padw, s=s, radius=pincer_r, **kwargs)
        CPW_straight(chip, s_right, length=pincer_l - s-pad_r, **kwargs)
        CPW_stub_open(chip, s_right, w=pincer_padw, s=s, **kwargs)
    else:
        s_left = s_left.cloneAlong(vector=(0,pincer_padw/2+s/2))
        Strip_bend(chip, s_left, CCW=True, w=s, radius=pincer_r + pincer_padw/2 - s/2, **kwargs)
        s_left = s_left.cloneAlong(vector=(s/2,s/2), newDirection=-90)
        Strip_straight(chip, s_left, length=pad_r + pincer_padw/2, w=s)

        s_right = s_right.cloneAlong(vector=(0,-pincer_padw/2-s/2))
        Strip_bend(chip, s_right, CCW=False, w=s, radius=pincer_r + pincer_padw/2 - s/2, **kwargs)
        s_right = s_right.cloneAlong(vector=(s/2,-s/2), newDirection=90)
        Strip_straight(chip, s_right, length=pad_r + pincer_padw/2, w=s)



    if not pincer_flipped:
        s_start.shiftPos(pincer_padw+pincer_tee_r+2*s)
        struct().updatePos(s_start.getPos())
    else: 
        struct().updatePos(s_start.getPos(),angle=180)
        #struct.direction = s_start.direction + 180
        
def CPW_tee_stub(chip,structure,stub_length,stub_w,tee_r=0,outer_width=None,w=None,s=None,pincer_flipped=False,bgcolor=None,**kwargs):
    '''
    stub_length :    end-to-end length of stub pin (not counting gap) 
    stub_w :   pincer pad trace width
    pincer_tee_r :  radius of tee
    pad_r:          inside radius of pincer bends
    w:              original trace width
    s:              pincer trace gap
    pincer_flipped: equivalent to hflip
    '''
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if w is None:
        try:
            w = struct().defaults['w']
        except KeyError:
            w=0
            print('\x1b[33mw not defined in ',chip.chipID)
    if s is None:
        try:
            s = struct().defaults['s']
        except KeyError:
            print('\x1b[33ms not defined in ',chip.chipID,'!\x1b[0m')
        
    if not pincer_flipped: s_start = struct().clone()
    else:
        struct().shiftPos(stub_w+2*s,angle=180)
        #struct().direction += 180
        s_start = struct().clone()

    s_left, s_right = CPW_tee(chip, struct(), w=w, s=s, w1=stub_w, s1=s, radius=tee_r + s, **kwargs)

    CPW_straight(chip, s_left, length=(stub_length-w-2*s-stub_w)/2, **kwargs)
    CPW_stub_round(chip, s_left,**kwargs)
    CPW_straight(chip, s_right, length=(stub_length-w-2*s-stub_w)/2, **kwargs)
    CPW_stub_round(chip, s_right,**kwargs)

    if not pincer_flipped:
        s_start.shiftPos(stub_w+2*s)
        struct().updatePos(s_start.getPos())
    else: 
        struct().updatePos(s_start.getPos(),angle=180)
        #struct.direction = s_start.direction + 180
    
# ===============================================================================
# Airbridges (Lincoln Labs designs)
# ===============================================================================
def setupAirbridgeLayers(wafer:m.Wafer,BRLAYER='BRIDGE',RRLAYER='TETHER',brcolor=41,rrcolor=32):
    #add correct layers to wafer, and cache layer
    wafer.addLayer(BRLAYER,brcolor)
    wafer.BRLAYER=BRLAYER
    wafer.addLayer(RRLAYER,rrcolor)
    wafer.RRLAYER=RRLAYER

def Airbridge(
    chip, structure, cpw_w=None, cpw_s=None, xvr_width=None, xvr_length=None, rr_width=None, rr_length=None,
    rr_br_gap=None, rr_cpw_gap=None, shape_overlap=0, br_radius=0, clockwise=False, lincolnLabs=False, BRLAYER=None, RRLAYER=None, **kwargs):
    """
    Define either cpw_w and cpw_s (refers to the cpw that the airbridge goes across) or xvr_length.
    xvr_length overrides cpw_w and cpw_s.
    """
    assert lincolnLabs, 'Not implemented for normal usage'
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if cpw_w is None:
        try:
            cpw_w = struct().defaults['w']
        except KeyError:
            print('\x1b[33mw not defined in ',chip.chipID)
    if cpw_s is None:
        try:
            cpw_s = struct().defaults['s']
        except KeyError:
            print('\x1b[33ms not defined in ',chip.chipID)

    #get layers from wafer
    if BRLAYER is None:
        try:
            BRLAYER = chip.wafer.BRLAYER
        except AttributeError:
            setupAirbridgeLayers(chip.wafer)
            BRLAYER = chip.wafer.BRLAYER
    if RRLAYER is None:
        try:
            RRLAYER = chip.wafer.RRLAYER
        except AttributeError:
            setupAirbridgeLayers(chip.wafer)
            RRLAYER = chip.wafer.RRLAYER

    if lincolnLabs:
        rr_br_gap = 1.5 # RR.BR.E.1
        if rr_cpw_gap is None: rr_cpw_gap = 2 # LL requires >= 0 (RR.E.1)
        else: assert rr_cpw_gap + rr_br_gap >= 1.5 # RR.E.1

        if xvr_length is None:
            xvr_length = cpw_w + 2*cpw_s + 2*(rr_cpw_gap)

        if 5 <= xvr_length <= 16: # BR.W.1, RR.L.1
            xvr_width = 5
            rr_length = 8
        elif 16 < xvr_length <= 27: # BR.W.2, RR.L.2
            xvr_width = 7.5
            rr_length = 10
        elif 27 < xvr_length <= 32: # BR.W.3, RR.L.3
            xvr_width = 10
            rr_length = 14
        rr_width = xvr_width + 3 # RR.W.1
        shape_overlap = 0.1 # LL requires >= 0.1
        delta = 0
        if br_radius > 0:
            r = br_radius - cpw_w/2 - cpw_s
            delta = r*(1-math.sqrt(1-1/r**2*((rr_width + 2*rr_br_gap)/2)**2))
        # this code does not check if your bend is super severe and the necessary delta
        # changes the necessary xvr_widths and rr_lengths, so don't do anything extreme

    if clockwise:
        delta_left = 0
        delta_right = delta
    else:
        delta_right = 0
        delta_left = delta

    s_left = struct().clone()
    s_left.direction += 90
    s_left.shiftPos(-shape_overlap)
    Strip_straight(chip, s_left, length=xvr_length/2+delta_left+2*shape_overlap, w=xvr_width, layer=BRLAYER, **kwargs)
    s_left.shiftPos(-shape_overlap)
    Strip_straight(chip, s_left, length=rr_length + 2*rr_br_gap, w=rr_width + 2*rr_br_gap, layer=BRLAYER, **kwargs)
    s_l = s_left.clone()
    s_left.shiftPos(-rr_length - rr_br_gap)
    Strip_straight(chip, s_left, length=rr_length, w=rr_width, layer=RRLAYER, **kwargs)

    s_right = struct().clone()
    s_right.direction -= 90
    s_right.shiftPos(-shape_overlap)
    Strip_straight(chip, s_right, length=xvr_length/2+delta_right+2*shape_overlap, w=xvr_width, layer=BRLAYER, **kwargs)
    s_right.shiftPos(-shape_overlap)
    Strip_straight(chip, s_right, length=rr_length + 2*rr_br_gap, w=rr_width + 2*rr_br_gap, layer=BRLAYER, **kwargs)
    s_r = s_right.clone()
    s_right.shiftPos(-rr_length - rr_br_gap)
    Strip_straight(chip, s_right, length=rr_length, w=rr_width, layer=RRLAYER, **kwargs)

    return s_l, s_r


def CPW_bridge(chip, structure, xvr_length=None, w=None, s=None, lincolnLabs=False, BRLAYER=None, RRLAYER=None, **kwargs):
    """
    Draws an airbridge to bridge two sections of CPW, as well as the necessary connections.
    w, s are for the CPW we want to connect.
    structure is oriented at the same place as the structure for Airbridge.
    """
    assert lincolnLabs, 'Not implemented for normal usage'
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if w is None:
        try:
            w = struct().defaults['w']
        except KeyError:
            print('\x1b[33mw not defined in ',chip.chipID)
    if s is None:
        try:
            s = struct().defaults['s']
        except KeyError:
            print('\x1b[33ms not defined in ',chip.chipID)

    if lincolnLabs:
        rr_br_gap = 1.5 # RR.BR.E.1
        rr_cpw_gap = 0 # LL requires >= 0 (RR.E.1)
        if xvr_length is None:
            xvr_length = w + 2*s + 2*(rr_cpw_gap)
        if 5 <= xvr_length <= 16:
            xvr_width = 5
            rr_length = 8
        elif 16 < xvr_length <= 27:
            xvr_width = 7.5
            rr_length = 10
        elif 27 < xvr_length <= 32:
            xvr_width = 10
            rr_length = 14
        else:
            assert False, f'xvr_length {xvr_length} is out of range'
        rr_width = xvr_width + 3

    s_left, s_right = Airbridge(chip, struct(), xvr_length=xvr_length, lincolnLabs=lincolnLabs, **kwargs)

    s_left.shiftPos(-rr_length - 2*rr_br_gap - rr_cpw_gap)
    CPW_straight(chip, s_left, length=rr_length + 2*rr_br_gap + rr_cpw_gap, w=rr_width + 2*rr_br_gap, s=s, **kwargs)
    CPW_taper(chip, s_left, length=rr_length + 2*rr_br_gap, w0=rr_width+2*rr_br_gap, s0=s, w1=w, s1=s, **kwargs)

    s_right.shiftPos(-rr_length - 2*rr_br_gap - rr_cpw_gap)
    CPW_straight(chip, s_right, length=rr_length + 2*rr_br_gap + rr_cpw_gap, w=rr_width + 2*rr_br_gap, s=s, **kwargs)
    CPW_taper(chip, s_right, length=rr_length + 2*rr_br_gap, w0=rr_width + 2*rr_br_gap, s0=s, w1=w, s1=s, **kwargs)

    return s_left, s_right
