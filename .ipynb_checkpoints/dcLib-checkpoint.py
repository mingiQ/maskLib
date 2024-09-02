# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:29:02 2019

@author: Sasha

Library for drawing standard components for DC measurements (Four probe resistance bars, etc)
"""

import maskLib.MaskLib as m
from dxfwrite import DXFEngine as dxf
from dxfwrite import const
from maskLib.microwaveLib import Strip_straight, Strip_stub_open
from maskLib.microwaveLib import CPW_stub_open,CPW_stub_short,CPW_straight


# ===============================================================================
# resistance bar chip (default designed for 7x7, but fully configurable for other sizes)
# ===============================================================================

class Rbar(m.Chip):
    def __init__(self,wafer,chipID,layer,bar_width = 40,bar_length = 1500, pad_x = 1000, pad_y = 800, pad_sep = 1000, bar_offs = 500):
        m.Chip7mm.__init__(self,wafer,chipID,layer)

        self.add(dxf.rectangle((0,0),pad_x,self.height,bgcolor=wafer.bg(),layer=wafer.defaultLayer))
        self.add(dxf.rectangle((self.width-pad_x,0),pad_x,self.height,bgcolor=wafer.bg(),layer=wafer.defaultLayer))
        self.add(dxf.rectangle((pad_x,0),self.width-2*pad_x,pad_y,bgcolor=wafer.bg(),layer=wafer.defaultLayer))
        self.add(dxf.rectangle((pad_x,self.height),self.width-2*pad_x,pad_y,valign=const.BOTTOM,bgcolor=wafer.bg(),layer=wafer.defaultLayer))
        
        self.add(dxf.rectangle(self.center,pad_sep,self.height - 2*pad_y,halign=const.CENTER,valign=const.MIDDLE,bgcolor=wafer.bg(),layer=wafer.defaultLayer))
        #inner
        if bar_offs > bar_width/2:
            #right
            self.add(dxf.rectangle(self.centered((bar_offs,0)),bar_offs - bar_width/2,bar_length,valign=const.MIDDLE,bgcolor=wafer.bg(),layer=wafer.defaultLayer))
            #left
            self.add(dxf.rectangle(self.centered((-bar_offs,0)),bar_offs - bar_width/2,bar_length,halign=const.RIGHT,valign=const.MIDDLE,bgcolor=wafer.bg(),layer=wafer.defaultLayer))
        else:
            bar_offs = bar_width/2
            print('Warning: bar offset too low in Rbar. Adjusting offset')
        #outer
        self.add(dxf.rectangle(self.centered((bar_offs + pad_sep/2 +bar_width/2,0)),self.width/2 - bar_offs - pad_x - pad_sep/2 - bar_width/2,bar_length,valign=const.MIDDLE,bgcolor=wafer.bg(),layer=wafer.defaultLayer))
        self.add(dxf.rectangle(self.centered((-bar_offs - pad_sep/2 -bar_width/2,0)),self.width/2 - bar_offs - pad_x - pad_sep/2 - bar_width/2,bar_length,halign=const.RIGHT,valign=const.MIDDLE,bgcolor=wafer.bg(),layer=wafer.defaultLayer))
        
def ResistanceBarBilayer(chip,structure,length=1500,width=40,pad=600,gap=50,r_out=None,secondlayer='SECONDLAYER',bgcolor=None):
    #write a resistance bar centered on the structure or position specified. Defaults to pointing in direction of current structure
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if bgcolor is None:
        bgcolor = chip.wafer.bg()
    if r_out is None:
        r_out = gap
    
    struct().shiftPos(-length/2-pad-gap)
    srBar=struct().clone(defaults={'w':pad,'s':gap,'r_out':gap})
    Strip_stub_open(chip,srBar,flipped=True,w=pad+2*gap)
    srBar2 = srBar.cloneAlong()
    Strip_straight(chip,srBar,pad,w=pad+2*gap)
    Strip_stub_open(chip,srBar,flipped=False,w=pad+2*gap)
    Strip_straight(chip, srBar, length-2*gap, w=width+2*gap)
    Strip_stub_open(chip,srBar,flipped=True,w=pad+2*gap)
    Strip_straight(chip,srBar,pad,w=pad+2*gap)
    Strip_stub_open(chip,srBar,w=pad+2*gap)
    
    Strip_straight(chip,srBar2,pad,w=pad,layer=secondlayer)
    Strip_straight(chip, srBar2, length, w=width,layer=secondlayer)
    Strip_straight(chip,srBar2,pad,w=pad,layer=secondlayer)
    
def ResistanceBarNegative(chip,structure,length=1500,width=40,pad=600,gap=50,r_out=None,bgcolor=None):
    #write a resistance bar centered on the structure or position specified. Defaults to pointing in direction of current structure
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if bgcolor is None:
        bgcolor = chip.wafer.bg()
    if r_out is None:
        r_out = gap
    
    struct().shiftPos(-length/2-pad-gap)
    srBar=struct().clone(defaults={'w':pad,'s':gap,'r_out':r_out})
    CPW_stub_open(chip,srBar,flipped=True)
    CPW_straight(chip,srBar,pad)
    CPW_stub_short(chip,srBar,flipped=False,w=width,s=(pad+2*gap-width)/2,curve_ins=False)
    CPW_straight(chip, srBar, length-2*gap, w=width)
    CPW_stub_short(chip,srBar,flipped=True,w=width,s=(pad+2*gap-width)/2,curve_ins=False)
    CPW_straight(chip,srBar,pad)
    CPW_stub_open(chip,srBar)
    
def ResistanceBar(chip,structure,length=1500,width=40,pad=600,r_out=50,secondlayer='SECONDLAYER',bgcolor=None):
    #write a resistance bar centered on the structure or position specified. Defaults to pointing in direction of current structure
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if bgcolor is None:
        bgcolor = chip.wafer.bg()
    
    struct().shiftPos(-length/2-pad)
    srBar=struct().clone(defaults={'w':pad,'r_out':r_out})
    
    Strip_straight(chip,srBar,pad,w=pad,layer=secondlayer)
    Strip_straight(chip, srBar, length, w=width,layer=secondlayer)
    Strip_straight(chip,srBar,pad,w=pad,layer=secondlayer)


def Extended_JJ(chip,structure,J_length=0.15, J_width=2, J_width1=3, lead_d=0.3, r_out=50,secondlayer='SECONDLAYER',bgcolor=None):
    #write a resistance bar centered on the structure or position specified. Defaults to pointing in direction of current structure
    '''
    J_length = junction length, um
    J_width = junction width, um
    lead_width = lead width, um
    '''
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if bgcolor is None:
        bgcolor = chip.wafer.bg()
    
    #struct().shiftPos(-length/2-pad)
    #starting point: set the center of the JJ as (0,0)
    struct().translatePos(vector=(-J_width1-J_width/2, (lead_d+J_length)/2))
    srBar=struct().clone()
    
    Strip_straight(chip,srBar, J_width1+J_width, w=lead_d, layer=secondlayer)

    srBar.translatePos(vector=(-J_width, -(lead_d+J_length)))
    srBar1=srBar.clone()
    
    Strip_straight(chip,srBar1, J_width1+J_width, w=lead_d, layer=secondlayer)
    #Strip_straight(chip,srBar,pad,w=pad,layer=secondlayer)

def nw_JJ(chip,structure, J_length=0.15, J_width1=[5,1], J_width2 = [1, 1], lead_d=[0.05, 0.3], taper= [2,2], r_out=50,secondlayer='SECONDLAYER',bgcolor=None):
    #write a resistance bar centered on the structure or position specified. Defaults to pointing in direction of current structure
    '''
    J_length = junction length, um
    J_width1 = [left lead length, right lead length] um
    lead_d = [junction width (narrow), tapered lead (wide)] um
    taper = [left tapered length, right tapered length]
    '''
    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if bgcolor is None:
        bgcolor = chip.wafer.bg()
    
    #struct().shiftPos(-length/2-pad)
    #starting point: set the center of the JJ as (0,0)
    
    struct().translatePos(vector=(-J_width1[0]-taper[0]-J_length/2, 0))
    #struct.shiftPos(distance=0, angle=45, newDir=None)
    srBar=struct().clone()
    Strip_straight(self,srBar, J_width1[0], w=lead_d[1])
    Strip_taper(self, srBar, length=taper[0], w0=lead_d[1], w1=lead_d[0])
    Strip_straight(self,srBar, J_width2[0], w=lead_d[0])

    srBar.translatePos(vector=(J_length, 0))
    srBar1=srBar.clone()

    Strip_straight(self,srBar1, J_width2[1], w=lead_d[0])
    Strip_taper(self, srBar1, length=taper[1], w0=lead_d[0], w1=lead_d[1])
    Strip_straight(self,srBar1, J_width1[1], w=lead_d[1])

def split_gate(chip, structure, length=10, d = 1, w0=5, s0=2,  w1=9, s1=5, loop = [70, 30], r_out=50,secondlayer='SECONDLAYER',bgcolor=None):
   '''
    Split gate for confining the supercurrent channel: refer https://arxiv.org/abs/2408.08487 
    
   '''

    def struct():
        if isinstance(structure,m.Structure):
            return structure
        elif isinstance(structure,tuple):
            return m.Structure(chip,structure)
        else:
            return chip.structure(structure)
    if bgcolor is None:
        bgcolor = chip.wafer.bg()
        
    #offset=[0,6]
    self.add(SkewRect(struct().getPos((d,-w0/2)),length,s0,(0, s1-s0),s1,rotation=struct().direction,valign=const.TOP,edgeAlign=const.TOP,bgcolor=bgcolor),structure=struct(),offsetVector=(length+d, -w0/2-s0+s1/2))
    Strip_straight(self,struct(), loop[0]/2-length-d, w=s1)
    #print(struct.getPos())
    Strip_bend(self, struct(), angle=90, CCW=False, w=s1, radius=s1)
    #print(struct.getPos())
    Strip_straight(self, struct(), loop[1], w=s1) 
    Strip_bend(self, struct(), angle=90, CCW=False, w=s1, radius=s1)
    Strip_straight(self, struct(), loop[0], w=s1) 
    Strip_bend(self, struct(), angle=90, CCW=False, w=s1, radius=s1)
    Strip_straight(self, struct(), loop[1], w=s1) 
    Strip_bend(self, struct(), angle=90, CCW=False, w=s1, radius=s1)
    Strip_straight(self,struct(), loop[0]/2-length-d, w=s1)
    self.add(SkewRect(struct().getPos((0,-w0/2)),length,s1,(0, s1-s0),s0,rotation=struct().direction,valign=const.BOTTOM,edgeAlign=const.BOTTOM,bgcolor=bgcolor),structure=struct(),offsetVector=(length, -w0/2-s0+s1/2))
    #self.add(SkewRect(struct.getPos((0,w0/2)),length,s0,(offset[0],w1/2-w0/2+offset[1]),s1,rotation=struct.direction,valign=const.BOTTOM,edgeAlign=const.BOTTOM,bgcolor=bgcolor),structure=struct,offsetVector=(length+offset[0],offset[1]))


