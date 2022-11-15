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
    