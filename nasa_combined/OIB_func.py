
def haversine(origin, destination):
    # Source: http://www.platoscave.net/blog/2009/oct/5/calculate-distance-latitude-longitude-python/
    import math
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
                                                  * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(
        dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d


def calc_distance(lat, lon):
    import numpy as np
    
    dist = np.zeros((np.size(lon)))
    for i in range(2, np.size(lon)):
        dist[i] = dist[i - 1] + haversine([lat[i - 1], lon[i - 1]],
                                                  [lat[i], lon[i]])
    return dist


def fill_nan(a):
    from scipy import interpolate
    import numpy.ma as ma
    import numpy as np

    '''
    interpolate to fill nan values
    '''
    b = ma.filled(a, np.nan)
    inds = np.arange(b.shape[0])
    good = np.where(np.isfinite(b))
    f = interpolate.interp1d(inds[good], b[good], bounds_error=False)
    c = np.where(np.isfinite(b), b, f(inds))
    return c


def linearly_interpolate_nans(y):
    import numpy as np
    # Fit a linear regression to the non-nan y values

    # Create X matrix for linreg with an intercept and an index
    X = np.vstack((np.ones(len(y)), np.arange(len(y))))

    # Get the non-NaN values of X and y
    X_fit = X[:, ~np.isnan(y)]
    y_fit = y[~np.isnan(y)].reshape(-1, 1)

    # Estimate the coefficients of the linear regression
    beta = np.linalg.lstsq(X_fit.T, y_fit)[0]

    # Fill in all the nan values using the predicted coefficients
    y.flat[np.isnan(y)] = np.dot(X[:, np.isnan(y)].T, beta)
    return y


def catATM(atmdir, date_flight):
    import sys, glob, time

    suffix = '.csv'

    # Get icessn filenames that were passed as arguments TODO use dates to grab only certain files
    pattern = os.path.join(atmdir, 'ILATM2_' + date_flight + '*_smooth_*' + suffix)
    print(('ATM pattern: {}'.format(pattern)))
    try:
        # filenames = [f for f in infiles if f.__contains__('_smooth_') if f.endswith('_50pt.csv')]
        filenames = sorted(glob.glob(pattern))  # , key=alphanum_key)
        # filenames[0]
        print('First up is {0}'.format(filenames[0]))
    except:
        print(__doc__)
    # exit()
    output_filename = os.path.join(atmdir, 'ILATM2_' + date_flight + '_all' + suffix)
    print(('Extracting records TO {0}'.format(output_filename)))
    tiles = [0]
    # Open output file
    with open(output_filename, 'w') as f:
        # Loop through filenames
        for filename in filenames:
            print(('Now extracting records from {0}...'.format(filename)))
            # Get date from filename
            # date = '20' + filename[:6]	# this is Linky's original code
            date = os.path.basename(filename)[7:15]
            prevTime = 0
            # Loop through lines in icessn file
            for line in open(filename):
                # Make sure records have the correct number of words (11)
                if (len(line.split()) == 11) and (int(line.split()[-1]) in tiles):
                    line = line.strip()
                    # gpsTime = float(line.split()[0])
                    gpsTime = float(line.split(',')[0])
                    # If seconds of day roll over to next day
                    if gpsTime < prevTime:
                        date = str(int(date) + 1)
                    # Create new data record (with EOL in "line" variable)
                    newline = '{0}, {1}'.format(date, line)
                    # print newline
                    f.write(newline + '\n')


def calc_icebase(sfc, thick):
    """
    :param sfc:
    :param thick:
    :return:
    // C0 = icebase_recalc
    // C1 = surface_recalc
    // C2 = THICK_redo
    @var1 = (C1 == DUMMY) ? (DUMMY): (C1 - C2);
    C0 = (@ var1 >= C1) ? (C1): (@ var1);
    """
    # icebase = sfc - thick
    # icebase = np.where(icebase >= 0, sfc, 0)
    icebase = np.where((sfc - thick) >= 0, sfc, sfc - thick)
    return icebase


def calc_surface(atm, radar):
    """
    :param atm:
    :param radar:
    :return:
    //C0=surface_recalc
    //C1=SURFACE_atm_redo
    //C2=TOPOGRAPHY_radar
    C0 = (C1 != DUMMY) ? (C1) : (C2);
    """
    surface = np.where(np.isnan(radar), atm, radar)
    return surface


def get_closest_cell_xr(ds, lat, lon, lat_key='lat', lon_key='lon'):
    """
    SSIA
    :param file_for_latlon:
    :param lat:
    :param lon:
    :return:
    """
    a = abs(ds[lon_key][:] - lon) + abs(ds[lat_key][:] - lat)
    iii, jjj = np.unravel_index(a.argmin(), a.shape)
    return iii, jjj


def get_closest_cell_llz(df_llz, lat, lon, lat_key='lat', lon_key='lon'):
    """
    SSIA
    :param df_llz:
    :param lat:
    :param lon:
    :return:
    """
    lat_llz = df_llz[lat_key][:]
    lon_llz = df_llz[lon_key][:]
    #     print('lat,lon', lat,lon)
    #     a = abs(LON - lon)
    #     print(a.min())
    #     print('Closest lat,lon', LAT[a.idxmin()], LON[a.idxmin()])

    #     a = abs(LAT - lat)
    #     print(a.min())
    #     print('Closest lat,lon', LAT[a.idxmin()], LON[a.idxmin()])

    a = abs(lon_llz - lon) + abs(lat_llz - lat)
    #     print(a.min())
    #     print('Closest lat,lon', LAT[a.idxmin()], LON[a.idxmin()])
    return a.idxmin()


def importOIBrad(basedir, timedir, infile):
    """
    :param basedir:
    :param timedir:
    :return:
    """
    datadir = 'IRMCR2'
    # infile = '2009_Antarctica_DC8'
    suffix = '.csv'

    ### Read ascii file as csv
    # headers = ('LAT','LONG','DATE','DOY','TIME','FLT','PSX','PSY','WGSHGT','FX','FY','FZ','EOTGRAV','FACOR','INTCOR','FAG070','FAG100','FAG140','FLTENVIRO')
    df = pd.read_csv(os.path.join(basedir, datadir, timedir, infile + suffix),
                     delimiter=",", na_values='-9999.00')
    # df.replace('-9999.00', np.nan)
    df.rename(columns={'SURFACE': 'SURFACE_radar'}, inplace=True)

    ### do some DATETIME operations
    # df['DATE'] = str(df['FRAME'][0])[:8]
    df['FRAMESTR'] = df['FRAME'].apply(str)
    df['DATE'] = pd.to_datetime(list(df.FRAMESTR.str[:8]), format='%Y%m%d')
    del df['FRAMESTR']
    df['UNIX'] = df['DATE'].astype(np.int64) // 10 ** 9
    df['UNIX'] = df['UNIX'] + df['TIME']
    df['iunix'] = pd.to_datetime(df['UNIX'] * 10 ** 3, unit='ms')
    df = df.set_index('iunix')
    return df


def importOIBrad_all(raddir, date_flight):
    """
    :param basedir:
    :param timedir:
    :return:
    """
    from glob import glob
    # basedir = '/Users/dporter/Documents/data_local/OIB/OIB/'
    # datadir = 'IRMCR2'
    # infile = '2009_Antarctica_DC8'
    suffix = '.csv'
    pattern = os.path.join(raddir, '*'+date_flight+'*' + suffix)
    filenames = sorted(glob(pattern))  # , key=alphanum_key)
    filecounter = len(filenames)

    df_all = {}
    for fnum, filename in enumerate(filenames, start=0):
        # print "RADAR data file %i is %s" % (fnum, filename)
        df = pd.read_csv(filename, delimiter=",", na_values='-9999.00')
        df.rename(columns={'SURFACE': 'SURFACE_radar'}, inplace=True)

        ### do some DATETIME operations
        # df['DATE'] = str(df['FRAME'][0])[:8]
        df['FRAMESTR'] = df['FRAME'].apply(str)
        df['DATE'] = pd.to_datetime(list(df.FRAMESTR.str[:8]), format='%Y%m%d')
        del df['FRAMESTR']
        df['UNIX'] = df['DATE'].astype(np.int64) // 10 ** 9
        try:
            df['UNIX'] = df['UNIX'] + df['TIME']  # df['UTCTIMESOD']
        except KeyError:
            df['UNIX'] = df['UNIX'] + df['UTCTIMESOD']  # TODO: columns changed after 2016
        df['iunix'] = pd.to_datetime(df['UNIX'] * 10 ** 3, unit='ms')
        df = df.set_index('iunix')
        if fnum == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])
    return df_all


def importOIBatm(atmdir, date_flight):
    """
    :param basedir:
    :param timedir:
    :return:
    """
    # datadir = 'ILATM2'
    prefix = 'ILATM2_'
    suffix = '.csv'
    infile = os.path.join(atmdir, prefix + date_flight + '_all' + suffix)

    ### Read ascii file as csv
    headers = (
        'DATE', 'TIME', 'LAT', 'LON', 'SURFACE_atm', 'SLOPESN', 'SLOPEWE', 'RMS', 'NUMUSED', 'NUMOMIT', 'DISTRIGHT',
        'TRACKID')
    df = pd.read_csv(infile, header=None)  # delimiter=r"\s+",
    df.rename(columns=dict(list(zip(df.columns, headers))), inplace=True)
    # del df['TIME2']

    ### do some DATETIME operations
    df['DATETIME'] = (df.DATE * 1e5) + df.TIME
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    df['UNIX'] = df['DATE'].astype(np.int64) // 10 ** 9
    df['UNIX'] = df['UNIX'] + df['TIME']
    df['iunix'] = pd.to_datetime(df['UNIX'] * 10 ** 3, unit='ms')
    df = df.set_index('iunix')
    return df


def importOIBgrav(gravdir, timedir, date_flight):
    """
    :param basedir:
    :param timedir:
    :return:
    """
    from glob import glob
    # datadir = 'IGGRV1B/temp'
    # infile = 'IGGRV1B_20091104_13100500_V016'
    # infile = 'IGGRV1B_20091031_11020500_V016'
    # infile = 'IGGRV1B_20091116_15124500_V016'
    suffix = '.txt'
    pattern = os.path.join(gravdir, timedir, 'IGGRV1B_'+date_flight+'*' + suffix)
    print(('gravity file pattern: {}'.format(pattern)))
    infile = sorted(glob(pattern))  # , key=alphanum_key)

    ### Read ascii file as csv
    # metadata ends on line 69, column names on line 70
    headers = (
        'LAT', 'LONG', 'DATE', 'DOY', 'TIME', 'FLT', 'PSX', 'PSY', 'WGSHGT', 'FX', 'FY', 'FZ', 'EOTGRAV', 'FACOR',
        'INTCOR',
        'FAG070', 'FAG100', 'FAG140', 'FLTENVIRO')
    # print "Reading gravity file: %s" % infile[0] + suffix %TODO why did I think this would be a list?
    print("Reading gravity file: %s" % infile[0] + suffix)
    df = pd.read_csv(infile[0], delimiter=r"\s+", header=None, names=headers, skiprows=70)
    # headers = df.columns[1:df.shape[1]]
    # df.rename(columns=dict(zip(df.columns,headers)), inplace=True)
    # df.rename(columns={'LONG': 'LON'}, inplace=True)
    # df['ENVIRO'] = df.columns[[19]]
    # df.drop(df.columns['FLTENVIRO'],axis=1,inplace=True)

    ### do some DATETIME operations
    df['DATETIME'] = (df.DATE * 1e5) + df.TIME
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    df['UNIX'] = df['DATE'].astype(np.int64) // 10 ** 9
    df['UNIX'] = df['UNIX'] + df['TIME']
    df['iunix'] = pd.to_datetime(df['UNIX'] * 10 ** 3, unit='ms')
    df.drop(['DATETIME'], axis=1, inplace=True)
    df = df.set_index('iunix')
    return df
