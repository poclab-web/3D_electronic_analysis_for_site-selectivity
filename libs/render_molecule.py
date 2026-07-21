"""Add model-coordinate guides to a 3Dmol-compatible molecular viewer.

The helpers mutate a viewer exposing the ``addCylinder``, ``addArrow``,
``addLabel``, and ``addSphere`` methods used by py3Dmol/3Dmol.js.  Display
coordinates and radii are in angstrom.  The mesh extents reproduce the model's
molecular frame after converting its two-bohr grid spacing with
``1 bohr = 0.529177 angstrom``.
"""

import itertools
import numpy as np

def addmesh(xyzview):
    """Draw the model grid's box edges and positive boundary-face guides.

    ``xyzview`` must be a mutable 3Dmol-compatible viewer.  The fixed displayed
    ranges are ``x=(-12, 6)``, ``y=(-6, 6)``, and ``z=(-12, 12)`` bohr,
    converted to angstrom; face lines are spaced by 2 bohr.  Drawing happens
    in place and the function intentionally has no return value.
    """
    radius=0.003
    step=0.529177*2
    rang=np.array([[-12,6],[-6,6],[-12,12]])*0.529177
    def addcylinder_func(rang,xyzview,radius):
        """Add one cylinder spanning the supplied x/y/z endpoint ranges."""
        xyzview.addCylinder({"start": {"x": rang[0][0], "y": rang[1][0], "z": rang[2][0]},
                             "end": {"x": rang[0][1], "y": rang[1][1], "z": rang[2][1]},
                             "radius": radius, "color": "gray"})
        return xyzview
    if False:#内部のグリッド表示
        for x,y in itertools.product(range(rang[0][0]+1,rang[0][1]),range(rang[1][0]+1,rang[1][1])):
            xyzview=addcylinder_func([[x,x],[y,y],rang[2]],xyzview,radius)
        for y,z in itertools.product(range(rang[1][0]+1,rang[1][1]),range(rang[2][0]+1,rang[2][1])):
            xyzview = addcylinder_func([rang[0], [y, y], [z,z]], xyzview, radius)
        for z,x in itertools.product(range(rang[2][0]+1,rang[2][1]),range(rang[0][0]+1,rang[0][1])):
            xyzview = addcylinder_func([[x,x], rang[1], [z, z]], xyzview, radius)

    if True:#エッジのグリッド表示

        for x, y in itertools.product(rang[0],rang[1]):
            xyzview = addcylinder_func([[x, x], [y, y], rang[2]], xyzview, radius * 5)

        for y,z in itertools.product(rang[1], rang[2]):
            xyzview = addcylinder_func([rang[0], [y, y], [z,z]], xyzview, radius*5)

        for z,x in itertools.product(rang[2], rang[0]):
            xyzview = addcylinder_func([[x,x], rang[1], [z, z]], xyzview, radius*5)

    if True: #面のグリッド表示
        l=list(itertools.product(rang[0],np.arange(rang[1][0],rang[1][1]+step,step)))+\
          list(itertools.product(np.arange(rang[0][0],rang[0][1]+step,step),rang[1]))
        for x, y in l:
            if x==rang[0][1] or y==rang[1][1]:
                xyzview = addcylinder_func([[x, x], [y, y], rang[2]], xyzview, radius * 2)

        l=list(itertools.product(rang[1],np.arange(rang[2][0],rang[2][1]+step,step)))+\
          list(itertools.product(np.arange(rang[1][0],rang[1][1]+step,step),rang[2]))
        for y,z in l:
            if y == rang[1][1] or z == rang[2][1]:
                xyzview = addcylinder_func([rang[0], [y, y], [z,z]], xyzview, radius*2)

        l=list(itertools.product(rang[2],np.arange(rang[0][0],rang[0][1]+step,step)))+\
          list(itertools.product(np.arange(rang[2][0],rang[2][1]+step,step),rang[0]))
        for z,x in l:
            if z == rang[2][1] or x == rang[0][1]:
                xyzview = addcylinder_func([[x,x], rang[1], [z, z]], xyzview, radius*2)



def addxyzarrow(xyzview):
    """Draw labelled Cartesian axes spanning -4 to +4 angstrom.

    ``xyzview`` must support the 3Dmol ``addArrow`` and ``addLabel`` methods.
    The viewer is mutated in place and the function intentionally returns
    ``None``.
    """
    d=4
    radius=0.025
    color="gray"
    xyzview.addArrow(
        {"start": {"x": -d, "y": 0, "z": 0},
         "end": {"x": d, "y": 0, "z": 0}, "radius": radius,
         "radiusRatio": 4, "mid": 0.9, "color": color})
    xyzview.addArrow(
        {"start": {"x": 0, "y": -d, "z": 0},
         "end": {"x": 0, "y": d, "z": 0}, "radius": radius,
         "radiusRatio": 4, "mid": 0.9, "color": color})
    xyzview.addArrow(
        {"start": {"x": 0, "y": 0, "z": -d},
         "end": {"x": 0, "y": 0, "z": d}, "radius": radius,
         "radiusRatio": 4, "mid": 0.9, "color": color})
    xyzview.addLabel("x", {"position": {"x": d, "y": 0, "z": 0},
                           "backgroundColor": color, "backgroundOpacity": 0.5})
    xyzview.addLabel("y", {"position": {"x": 0, "y": d, "z": 0},
                           "backgroundColor": color, "backgroundOpacity": 0.5})
    xyzview.addLabel("z", {"position": {"x": 0, "y": 0, "z": d},
                           "backgroundColor": color, "backgroundOpacity": 0.5})

def add_label(xyz,param,center):
    """Draw symmetric geometric annotations around a molecular-frame point.

    Parameters
    ----------
    xyz
        Mutable 3Dmol-compatible viewer.
    param
        Three-element sequence ``(radial_span, arc_diameter, angle_deg)``.
        Length-like values are in angstrom and ``angle_deg`` controls the arc
        extent in degrees.
    center
        Three Cartesian coordinates in angstrom.  Markers are placed at
        ``(center_x, center_y, +/-center_z)`` and radial arrows use
        ``radial_span`` along ``x``.

    Notes
    -----
    The viewer is mutated in place; no value is returned.
    """
    radius=0.05
    radiusRatio=3
    color="black"
    xyz.addArrow({"start": {"x": 0, "y": 0, "z": 0},
                         "end": {"x": center[0], "y": 0, "z": center[2]},
                         "radius": radius, "color": color,"radiusRatio": radiusRatio,"mid": 0.8})
    xyz.addArrow({"start": {"x": 0, "y": 0, "z": 0},
                     "end": {"x": center[0], "y": 0, "z": -center[2]},
                     "radius": radius, "color": color,"radiusRatio": radiusRatio,"mid": 0.8})
    # xyz.addCylinder({"start": {"x": 0, "y": 0, "z": 0},
    #                  "end": {"x": param[1], "y": 0, "z": 0},
    #                  "radius": radius, "color": color})

    xyz.addArrow({"start": {"x": center[0]+param[0]/2,"y":center[1],"z":center[2]},
                  "end": {"x": center[0]+param[0],"y":center[1],"z":center[2]},
                  "radius": radius, "color": color, "radiusRatio": radiusRatio, "mid": 0.5})
    xyz.addArrow({"start": {"x": center[0]+param[0]/2, "y": center[1], "z": center[2]},
                  "end": {"x": center[0], "y": center[1], "z": center[2]},
                  "radius": radius, "color": color, "radiusRatio": radiusRatio, "mid": 0.5})
    xyz.addArrow({"start": {"x": center[0] + param[0] / 2, "y": center[1], "z": -center[2]},
                  "end": {"x": center[0] + param[0], "y": center[1], "z": -center[2]},
                  "radius": radius, "color": color, "radiusRatio": radiusRatio, "mid": 0.5})
    xyz.addArrow({"start": {"x": center[0] + param[0] / 2, "y": center[1], "z": -center[2]},
                  "end": {"x": center[0], "y": center[1], "z": -center[2]},
                  "radius": radius, "color": color, "radiusRatio": radiusRatio, "mid": 0.5})
    """
    xyz.addCurve({"points": [{"x":center[0],"y":center[1],"z":center[2]}, {"x":center[0]+param[0],"y":center[1],"z":center[2]}],
                              "radius":0.1,
                              "fromArrow":True,
                              "toArrow": True,
                              "color":'orange',
                              })
                              """
    for t in np.arange(0,param[2],1):
        xyz.addCylinder({"start": {"x": param[1]/2*np.cos(np.radians(t)), "y": 0, "z": param[1]/2*np.sin(np.radians(t))},
                             "end": {"x": param[1]/2*np.cos(np.radians(t+1.1)), "y": 0, "z": param[1]/2*np.sin(np.radians(t+1.1))},
                             "radius": radius, "color": color})
        xyz.addCylinder(
            {"start": {"x": param[1]/2 * np.cos(np.radians(t)), "y": 0, "z": -param[1]/2 * np.sin(np.radians(t))},
             "end": {"x": param[1]/2 * np.cos(np.radians(t + 1.1)), "y": 0, "z": -param[1]/2 * np.sin(np.radians(t + 1.1))},
             "radius": radius, "color": color})
    """xyz.addArrow({"start": {"x": param[1]*np.cos(np.radians(param[2]-10)), "y": 0, "z": param[1]*np.sin(np.radians(param[2]-10))},
                  "end": {"x": param[1]*np.cos(np.radians(param[2])), "y": 0, "z": param[1]*np.sin(np.radians(param[2]))},
                  "radius": radius, "color": "blue", "radiusRatio": 2, "mid": 0})"""

    """xyz.addLabel("{} [deg.]".format("θ"), {"position": {"x": param[1]*np.cos(np.radians(param[2]/2)), "y": 0, "z": param[1]*np.sin(np.radians(param[2]/2))},
                           "backgroundColor": "black", "backgroundOpacity": 0.5})
    xyz.addLabel("{} [deg.]".format("-θ"), {"position": {"x": param[1] * np.cos(np.radians(param[2] / 2)), "y": 0,
                                                        "z": -param[1] * np.sin(np.radians(param[2] / 2))},
                                           "backgroundColor": "black", "backgroundOpacity": 0.5})
    xyz.addLabel("{} [Å]".format("d"), {"position": {"x": param[1] * np.cos(np.radians(param[2]))/2, "y": 0,
                                                        "z": param[1] * np.sin(np.radians(param[2]))/2},
                                           "backgroundColor": "black", "backgroundOpacity": 0.5})
    xyz.addLabel("{} [Å]".format("d"), {"position": {"x": param[1] * np.cos(np.radians(param[2]))/2, "y": 0,
                                                         "z": -param[1] * np.sin(np.radians(param[2]))/2},
                                            "backgroundColor": "black", "backgroundOpacity": 0.5})
    xyz.addLabel("{} [Å]".format("r"), {"position": {"x": center[0]+param[0]/2,"y":center[1],"z":center[2]},
                                        "backgroundColor": "black", "backgroundOpacity": 0.5})"""
    xyz.addSphere(
        {"center": {"x": center[0], "y": center[1], "z": center[2]}, 'opacity': 1, "radius": radius * 2,
         "color": "black"})
    xyz.addSphere(
        {"center": {"x": center[0], "y": center[1], "z": -center[2]}, 'opacity': 1, "radius": radius*2,
         "color": "black"})
