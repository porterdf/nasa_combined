def calc_twtt(H, c):
    # PickDepth = -(.5*cIce)*PickTime
    # SurfDepth = (.5*cAir)*SurfTime
    
    twtt = H / (.5 * c )
    
    return twtt


def read_DICE_matfile(infile):
    from scipy.io import loadmat
    import pandas as pd

    ## Constants
    cAir = 299792458
    cIce = 1.68e8
    
    ## Load mat file
    matdata = loadmat(infile)
    
    # dict_keys(['__header__', '__version__', '__globals__', 'pickername', 'NOTE_VertScale', 'Time', 'Surf_Elev', 'GPS_time', 'FlightElev', 'SurfTime', 'Lat', 'Lon', 'Pixel', 'PickTime', 'X', 'Y', 'Distance', 'xdisp', 'Depth', 'Bright', 'MultipleBright', 'NoiseFloor', 'Notes', 'Data', 'VertScale'])
    
    # Data = matdata['Data'][:].squeeze()
    X = matdata['X'][:].squeeze()*1e3
    Y = matdata['Y'][:].squeeze()*1e3
    lat = matdata['Lat'][:].squeeze()
    lon = matdata['Lon'][:].squeeze()

    FlightElev = matdata['FlightElev'][:].squeeze()
    SurfTime = matdata['SurfTime'][:].squeeze()  
    PickTime = matdata['PickTime'][:].squeeze()
    Surf_elev = matdata['Surf_Elev'][:].squeeze()
    Time = matdata['Time'][:].squeeze()
    Depth = matdata['Depth'][:].squeeze()
    
    GPS_time = matdata['GPS_time'][:].squeeze()
    Pixel = matdata['Pixel'][:].squeeze()  
    Distance = matdata['Distance'][:].squeeze()
    Bright = matdata['Bright'][:].squeeze()
    
    # Derive
    PickDepth = -(.5*cIce)*PickTime
    SurfDepth = (.5*cAir)*SurfTime
    
    # Write as dataframe
    df = pd.DataFrame(data=[GPS_time, Time,
                            X, Y, lat, lon,
                            PickTime, SurfTime, 
                            FlightElev, Surf_elev, Depth,
                            Bright, Distance,
                            PickDepth, SurfDepth]).T
    
    df.rename(columns={0: 'GPS_time', 1: 'Time',
                       2: 'EPSG_X', 3: 'EPSG_Y', 4: 'lat', 5: 'lon',
                       6: 'PickTime', 7: 'SurfTime', 
                       8: 'FlightElev', 9: 'Surf_elev', 10: 'Depth',
                       11: 'Bright', 12: 'Distance',
                       13: 'PickDepth', 14: 'SurfDepth'},
                inplace=True)
    
    return df


def print_raster(raster):
    print(
        f"shape: {raster.shape}\n"
#         f"resolution: {raster.resolution()}\n"
        f"bounds: {raster.bounds}\n"
#         f"sum: {raster.sum().item()}\n"
        f"CRS: {raster.crs}\n"
    )
    

def get_custom_cmap(shade=3):
    import numpy as np
    from matplotlib.colors import ListedColormap

    cmap = np.zeros([256, 4])
    cmap[:, shade] = np.linspace(0, 1, 256)
    cmap = ListedColormap(cmap)
    
    return cmap


def fix_PROJ_path(postfix='/share/proj'):
    import pyproj
    import sys

    projpath = sys.prefix + postfix
    pyproj.datadir.set_data_dir(projpath)



def calc_density_average(H_ice,rho_firn=0.7, H_firn=50):
    '''
    In meters and kg/m3
    '''
    rho_water = 1.025
    rho_ice = 0.917
    
    H_ave = H_ice.mean()
    rho_ave = (rho_ice * (H_ave - H_firn)/H_ave) + (rho_firn * H_firn/H_ave)
    
    return rho_ave


def calc_hydrostatic_thickness(freeboard, rho_ave=0.917):
    rho_water = 1.025
    
    # =h_geoid/(1-(rho_ave/1.035))
    H_hydro = freeboard / (1 - (rho_ave / rho_water))
    icebase_hydro = freeboard - H_hydro
    
    return icebase_hydro

    
def read_ROSETTA_csv(infile):
    import pandas as pd
    
    ## read into DataFrame
    df = pd.read_csv(infile)
    
    ## New datetime index
    df.index = pd.to_datetime(df["unixtime"], unit='s')
    
    ## Calculate some new variables
    df.loc[:, 'icebase_dice'] = df.rosetta_lidar - df.thickness_dice
    
    return df
    
    