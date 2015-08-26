import numpy as np

    # x,z components of each panel's tangential and normal vectors
    # Ns is the numper of panels in spanwise direction
    #Nc is the number of panels in chordwise direction
# TODO: panel_vectors now need a second tangential vector and area of panel
def panel_vectors(x,y,z,Nc,Ns):

    # each column stores another spanwise cros sectional panel lengths
    lpanel = np.sqrt((x[1:,0:Ns]-x[:-1,0:Ns])**2 + (y[1:,0:Ns]-y[:-1,0:Ns])**2 + (z[1:,0:Ns]-z[:-1,0:Ns])**2)
    tx = (x[1:,0:Ns]-x[:-1,0:Ns])/lpanel
    ty = (y[0:Nc,1:]-y[0:Nc,:-1])/lpanel.T
    tz = (z[1:,0:Ns]-z[:-1,0:Ns])/lpanel
    #nx = -tz
    #nz = tx
    return (tx,ty,tz,nx,ny,nz,lpanel)

    # x,z components of each midpoint's/collocation point's tangential and normal vectors
# TODO: point_vectors now need a second tangential vector
def point_vectors(v1,v2,v3,v4):
    vy = v1 - v2
    vx = v3 - v4        
    vn = np.cross(vx,vy) # normal vectors at every panel corner points
    
    #components of normal vectors
    vn_x = vn(1)/np.linalg.norm(vn)
    vn_y = vn(2)/np.linalg.norm(vn)
    vn_z = vn(3)/np.linalg.norm(vn)  
    
    return[vn_x,vn_y,vn_z]

#    tx = (xdp-xdm)/np.sqrt((xdp-xdm)**2 + (ydp-ydm)**2 + (zdp-zdm)**2)
#    ty = (ydp-ydm)/np.sqrt((xdp-xdm)**2 + (ydp-ydm)**2 + (zdp-zdm)**2)
#    tz = (zdp-zdm)/np.sqrt((xdp-xdm)**2 + (ydp-ydm)**2 + (zdp-zdm)**2)
##    nx = -tz
##    nz = tx
#    return(tx,ty,tz,nx,ny,nz)

def archive(array, axis=0):
    """
    Shifts array values along an axis (row-wise by default).

    Used for arrays that keep past values for differencing with respect to time.

    Args:
        array: The array that will be manipulated.
        axis: The axis to shift values along (0==row-wise, 1==column-wise).
    """
    if len(np.shape(array)) == 1:
        array[1:] = array[:-1]
    elif len(np.shape(array)) == 2:
        if axis == 0:
            array[1:,:] = array[:-1,:]
        else:
            array[:,1:] = array[:,:-1]

# Velocity and velocity potential equations are defined in panel coordinates so a transformation should be done
# Each row of xp1/xp2/zp is an influence, and each column is a target
# NI is N influences, NT is N targets
# xi/zi is x/z of influences, xt/zt is x/z of target points

# TODO: Determine if there needs to be a major change in transformation module
def transformation(xt,zt,xi,zi):
# Returns xp1, xp2, zp
# Others: NT, NI, tx, tz, nx, nz, dummy, x_tile, z_tile, tx_tile, tz_tile

    NT = np.size(xt)
    NI = np.size(xi)-1

    (tx,tz,nx,nz) = panel_vectors(xi,zi)[:-1]

    # Intermediary variables to reduce number of tile/repeat operations
    # From normalvectors: tx==nz, tz==-nx
    x_tile = np.repeat(xt[:,np.newaxis].T,NI,0) - np.repeat(xi[:-1,np.newaxis],NT,1)
    z_tile = np.repeat(zt[:,np.newaxis].T,NI,0) - np.repeat(zi[:-1,np.newaxis],NT,1)
    tx_tile = np.repeat(tx[:,np.newaxis],NT,1)
    tz_tile = np.repeat(tz[:,np.newaxis],NT,1)

    # Transforming left side collocation points from global to local coordinates
    xp1 = x_tile*tx_tile + z_tile*tz_tile
    zp = x_tile*(-tz_tile) + z_tile*tx_tile

    # Transforming right side panel points into local coordinate system
    dummy = (xi[1:]-xi[:-1])*tx + (zi[1:]-zi[:-1])*tz
    xp2 = xp1 - np.repeat(dummy[:,np.newaxis],NT,1)

    return(xp1,xp2,zp)