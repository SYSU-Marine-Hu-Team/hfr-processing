import numpy as np
import pandas as pd
import copy

from codarutils import get_columns, unique_rows
from hfradarpy.radials import Radial, qc_radial_file


class QCRadial(Radial):
    # adapted from https://github.com/teresaupdyke/qccodar/blob/main/src/qccodar/qcutils.py
    def threshold_qc_doa_peak_power(self, threshold=5.0):
        """Bad Flag any DOA peak power (dB) less than threshold value (default 5.0 dB).

        Flags any direction of arrival (DOA) peak power (dB) that falls
        below the input threshold value (default 5.0 dB).  Depending on
        the value of MSEL (1, 2, or 3), MSR1, MDR1, or MDR2 columns are
        evaluated.  Returns modified matrix with VFLG column the only
        changed values.

        """

        havenan = np.isnan(self.data['MSR1']) | np.isnan(self.data['MDR1']) | np.isnan(self.data['MDR2'])
        bad = (self.data['MSEL']==1) & (self.data['MSR1']<float(threshold))| \
            ((self.data['MSEL']==2) & (self.data['MDR1']<float(threshold))) | \
            ((self.data['MSEL']==3) & (self.data['MDR2']<float(threshold))) | havenan
        self.data.loc[bad, 'VFLG'] = self.data.loc[bad, 'VFLG'] + (1<<1)

    # adapted from https://github.com/teresaupdyke/qccodar/blob/main/src/qccodar/qcutils.py
    def threshold_qc_doa_half_power_width(self, threshold=50.0):
    #def threshold_qc_doa_half_power_width(d, types_str, threshold=50.0):
        """Bad Flag DOA 1/2 Power Width (degress) greater than threshold value (default 50.0 degrees).

        Flags any direction of arrival (DOA) 1/2 Power width (degress)
        that is wider than the input threshold value (default 50.0
        degrees).  Depending on the value of MSEL (1, 2, or 3), MSW1,
        MDW1, or MDW2 columns are evaluated.  Returns modified matrix with
        VFLG column the only changed values.

        """

        havenan = np.isnan(self.data['MSW1']) | np.isnan(self.data['MDW1']) | np.isnan(self.data['MDW2'])
        bad = (self.data['MSEL']==1) & (self.data['MSW1']>float(threshold))| \
            ((self.data['MSEL']==2) & (self.data['MDW1']>float(threshold))) | \
            ((self.data['MSEL']==3) & (self.data['MDW2']>float(threshold))) | havenan
        self.data.loc[bad, 'VFLG'] = self.data.loc[bad, 'VFLG'] + (1<<2)

    # adapted from https://github.com/teresaupdyke/qccodar/blob/main/src/qccodar/qcutils.py
    def threshold_qc_monopole_snr(self, threshold=5.0):
        """Bad flag any SNR on monopole (dB)  less than threshold value (default 5.0 dB).

        Flags any signal-to-noise ratio (SNR) on monopole (dB) that falls
        below the input threshold value (default 5.0 dB).  No dependency on MSEL selections.

        """
        bad = self.data['MA3S'] < float(threshold)
        self.data.loc[bad, 'VFLG'] = self.data.loc[bad, 'VFLG'] + (1<<3)

    # adapted from https://github.com/teresaupdyke/qccodar/blob/main/src/qccodar/qcutils.py
    def threshold_qc_loop_snr(self, threshold=5.0):
    #def threshold_qc_loop_snr(d, types_str, threshold=5.0):
        """Bad flag if both loop SNR are less than threshold value (default 5.0 dB).

        Flags if signal-to-noise ratio (SNR) (dB) on loop1 AND on loop2 falls
        below the input threshold value (default 5.0 dB). No dependency on MSEL selections.

        """
        bad = (self.data['MA1S']<float(threshold)) & (self.data['MA2S']<float(threshold))
        self.data.loc[bad, 'VFLG'] = self.data.loc[bad, 'VFLG'] + (1<<3)

    # adapted from https://github.com/teresaupdyke/qccodar/blob/main/src/qccodar/qcutils.py
    def threshold_qc_all(self, qccodar_values = dict()):
        """Combine all three threshold tests

        Returns modified matrix with VFLG column only changed values.

        """

        if self.is_valid():

            qc_keys = qccodar_values.keys()

            # run high frequency radar qartod tests on open radial file
            if 'qc_doa_peak_power' in qc_keys:
                self.threshold_qc_doa_peak_power(qccodar_values['qc_doa_peak_power']['doa_peak_power_min'])

            if 'qc_doa_half_power_width' in qc_keys:
                self.threshold_qc_doa_half_power_width(qccodar_values['qc_doa_half_power_width']['doa_half_power_width_max'] )

            if 'qc_monopole_snr' in qc_keys:
                self.threshold_qc_monopole_snr(qccodar_values['qc_monopole_snr']['monopole_snr_min'])

            if 'qc_loop_snr' in qc_keys:
                self.threshold_qc_loop_snr(qccodar_values['qc_loop_snr']['loop_snr_min'])

    # adapted from https://github.com/teresaupdyke/qccodar/blob/main/src/qccodar/qcutils.py
    def threshold_rsd_numpoints(self, radialshort_velocity_count_min=1):
        """Bad flag any radialshort data with doppler velocity count (EDVC) less than "numpoints"

        Returns modified rsd matrix with VFLG column only changed if EDVC
        count is less than or equal to numpoints.  This threshold is
        checked after weighted_velocities()

        """

        bad = self.data['EDVC']<int(radialshort_velocity_count_min)
        self.data.loc[bad, 'VFLG'] = self.data.loc[bad, 'VFLG'] + (1<<12)

    # adapted from https://github.com/teresaupdyke/qccodar/blob/main/src/qccodar/qcutils.py
    def weighted_velocities(self, numdegrees=3, weight_parameter='MP'):
        """Calculates weighted average of radial velocities (VELO) at bearing and range.

        The weighted average of velocities found at given range and
        bearing based on weight_parameter.
        
        NOTE:  this version, like an earlier version, uses numpy array for building the RadialShort data array
        Indexing into the numpy array to fill data within the loop is much, much faster
        
        %timeit df2 = weighted_velocities_np(rmqc, numdegrees=3, weight_parameter='MP')
        1.15 s per loop (mean of 7 runs, 1 loop each)
        
        Paramters
        ---------
        r : Radial object -- created by hfradarpy with the data from LLUV file(s). 
        types_str : string 
            The 'TableColumnTypes' string header of LLUV file(s) provide keys for each column.
        weight_parameter : string ('MP', 'SNR3', 'NONE'), optional 
            If 'MP' (default), uses MUSIC antenna peak power values for weighting function
            using MSEL to select one of (MSP1, MDP1, or MDP2).
            If 'SNR3', uses signal-to-noise ratio on monopole (MA3S).
            If 'NONE', just average with no weighting performed.
        numdegrees: int, optional (default 3 degree)
        The number of degrees of bearing from which to get velocities to spatially average over.
        For example, 
            If 1 deg, velocities from window of 1 deg will be averaged.
            If 3 deg, velocities from a window of 3 degrees will be averaged. This is the default.
            If 5 deg, velocities from a window of 5 degrees will be averaged.

        Returns
        -------
        df : pandas Dataframe with columns labeled
        The averaged values with range and bearing.

        """

        # 
        # order of columns and labels for output data
        xcol_labels = ['VFLG', 'SPRC', 'BEAR', 'VELO', 'ESPC', 'MAXV', 'MINV', 'EDVC', 'ERSC']
        xc = get_columns(' '.join(xcol_labels))
        
        # data and columns from input Radial object as numpy array
        d = copy.deepcopy(self.data.to_numpy()) # NOTE using np array, not the dataframe
        c = get_columns( ' '.join(self.data.columns.to_list()) ) # dict of column labels, their index
        offset = ((numdegrees-1)/2)

        ud = unique_rows(d[:,[c['SPRC'],c['BEAR'],c['VFLG']]].copy())
        # return only rows that have VFLG==0 (0 == good, >0 bad) so only get good data
        ud = ud[ud[:,2]==0]
        if ud.size == 0:
            self.data = np.array([])
            return

        #
        allbearings = np.unique(ud[:,1])
        allranges = np.unique(ud[:,0])
        ud = np.array([[r, b] for r in allranges for b in allbearings])
        
        #
        nrows, _ = ud.shape
        ncols = len(xc)
        xd = np.ones(shape=(nrows,ncols))*np.nan
        #
        for irow, cell in enumerate(ud):
            rngcell, bearing = cell[0:2]
            # np.where() returns a tuple for condition so use np.where()[0]
            # also VFLG must equal 0 (0 == good, >0 bad) so only get good data
            xrow = np.where((d[:,c['SPRC']]==rngcell) & \
                            (d[:,c['BEAR']]>=bearing-offset) & \
                            (d[:,c['BEAR']]<=bearing+offset) & \
                            (d[:,c['VFLG']]==0))[0]
            # If no row matches rngcell AND bearing, then no VELO data, skip to next bearing
            if xrow.size == 0: 
                continue

            xcol = np.array([c['VELO'], c['MSEL'], c['MSP1'], c['MDP1'], c['MDP2'], c['MA3S']])
            a = d[np.ix_(xrow, xcol)].copy()

            # if xrow.size == edvc:
            VELO = a[:,0] # all radial velocities found in cell
            SNR3 = a[:,5] # SNR on monopole for each velocity
            if weight_parameter.upper() == 'MP':
                # Create array to hold each Music Power (based on MSEL)
                MP = np.array(np.ones(VELO.shape)*np.nan) 
                # pluck the msel-based Music Power from MSP1, MDP1 or MPD2 column
                for msel in [1, 2, 3]:
                    which = a[:,1]==msel
                    MP[which,] = a[which, msel+1]
                # convert MP from db to voltage for weighting
                MP = np.power(10, MP/10.)
                wts = MP/MP.sum()
                velo = np.dot(VELO,wts)
            elif weight_parameter.upper() == 'SNR3' or weight_parameter.upper() == 'SNR':
                wts = SNR3/SNR3.sum()
                velo = np.dot(VELO,wts)
            elif weight_parameter.upper() == 'NONE':
                # do no weighting and just compute the mean of all velo's
                velo = VELO.mean()
            # data
            xd[irow,xc['VFLG']] = 0
            xd[irow,xc['SPRC']] = rngcell
            xd[irow,xc['BEAR']] = bearing
            xd[irow,xc['VELO']] = velo
            # other stat output
            xd[irow,xc['ESPC']] = VELO.std() # ESPC
            xd[irow,xc['MAXV']] = VELO.max() # MAXV
            xd[irow,xc['MINV']] = VELO.min() # MINV
            # (EDVC and ERSC are the same in this subroutine's context)
            xd[irow,xc['EDVC']] = VELO.size # EDVC Velocity Count 
            xd[irow,xc['ERSC']] = VELO.size # ERSC Spatial Count

        # create dataframe with the np array
        df = pd.DataFrame(xd)
        df.columns = xcol_labels
        # delete extra lines (nan) not filled above
        df.dropna(axis=0, how='all', inplace=True)
        # ESPC had several nan's, replace these and other nan before returning?
        df.fillna(999.000, inplace=True)

        return df

    # adapted from https://github.com/teresaupdyke/qccodar/blob/main/src/qccodar/qcutils.py
    def deprecated_weighted_velocities_df(self, numdegrees=3, weight_parameter='MP'):
        """Calculates weighted average of radial velocities (VELO) at bearing and range.

        The weighted average of velocities found at given range and
        bearing based on weight_parameter.
        
        DATAFRAME VERSION -- is much slower -- takes 40-50 sec to do weighted velocities for one RadialShort

        %timeit df1 = weighted_velocities_df(rmqc, numdegrees=3, weight_parameter='MP')
        52.1 s per loop (mean of 7 runs, 1 loop each)

        Keeping here for documentation reasons
        
        Paramters
        ---------
        r : Radial object -- created by hfradarpy
        weight_parameter : string ('MP', 'SNR3', 'NONE'), optional 
            If 'MP' (default), uses MUSIC antenna peak power values for weighting function
            using MSEL to select one of (MSP1, MDP1, or MDP2).
            If 'SNR3', uses signal-to-noise ratio on monopole (MA3S).
            If 'NONE', just average with no weighting performed.
        numdegrees: int, optional (default 3 degree)
        The number of degrees of bearing from which to get velocities to spatially average over.
        For example, 
            If 1 deg, velocities from window of 1 deg will be averaged.
            If 3 deg, velocities from a window of 3 degrees will be averaged. This is the default.
            If 5 deg, velocities from a window of 5 degrees will be averaged.

        Returns
        -------
        xd : pandas Dataframe with columns labeled
        The averaged values with range and bearing.

        """
        print(" ... DATAFRAME VERSION OF WEIGHTED_VELOCITIES")
        # 
        # order of columns and labels for output data
        xcols = ['VFLG', 'SPRC', 'BEAR', 'VELO', 'ESPC', 'MAXV', 'MINV', 'EDVC', 'ERSC']

        d = copy.deepcopy(self.data)
        offset = ((numdegrees-1)/2)

        self.data = self.data.drop_duplicates(subset=['SPRC','BEAR','VFLG'])
        # return only rows that have VFLG==0 (0 == good, >0 bad) so only get good data
        good = self.data['VFLG'] == 0
        ud = self.data.loc[good, :]
        if ud.size == 0:
            self.data = np.array([])
            return

        allbearings = np.unique(ud['BEAR'])
        allranges = np.unique(ud['SPRC'])
        ud = np.array([[r, b] for r in allranges for b in allbearings])
        nrows, _ = ud.shape
        xd = pd.DataFrame(columns = xcols ,index=np.arange(nrows))


        for irow, cell in enumerate(ud):
            rngcell, bearing = cell[0:2]
            # np.where() returns a tuple for condition so use np.where()[0]
            # also VFLG must equal 0 (0 == good, >0 bad) so only get good data

            xrow = np.where((d['SPRC']==rngcell) & \
                            (d['BEAR']>=bearing-offset) & \
                            (d['BEAR']<=bearing+offset) & \
                            (d['VFLG']==0))[0]

            # If no row matches rngcell AND bearing, then no VELO data, skip to next bearing
            if xrow.size == 0: 
                continue

            #xcol = np.array([['VELO'], ['MSEL'], ['MSP1'], ['MDP1'], ['MDP2'], ['MA3S']])

            #a = d[np.ix_(xrow, xcol)].copy()
            ao = copy.deepcopy(d)
            a = ao.loc[xrow, ['VELO', 'MSEL', 'MSP1', 'MDP1', 'MDP2', 'MA3S']]
            # if xrow.size == edvc:
            VELO = a['VELO']  # all radial velocities found in cell
            SNR3 = a['MA3S']  # SNR on monopole for each velocity
            if weight_parameter.upper() == 'MP':
                # Create array to hold each Music Power (based on MSEL)
                MP = np.array(np.ones(VELO.shape)*np.nan) 
                # pluck the msel-based Music Power from MSP1, MDP1 or MPD2 column
                mselcol = ['','MSP1','MDP1','MDP2']
                for msel in [1, 2, 3]:
                    which = a['MSEL'] == msel
                    MP[which,] = a.loc[which, mselcol[msel]]
                # convert MP from db to voltage for weighting
                MP = np.power(10, MP/10.)
                wts = MP/MP.sum()
                velo = np.dot(VELO,wts)
            elif weight_parameter.upper() == 'SNR3' or weight_parameter.upper() == 'SNR':
                wts = SNR3/SNR3.sum()
                velo = np.dot(VELO,wts)
            elif weight_parameter.upper() == 'NONE':
                # do no weighting and just compute the mean of all velo's
                velo = VELO.mean()
            # data
            xd.loc[irow,['VFLG']] = 0
            xd.loc[irow,['SPRC']] = rngcell
            xd.loc[irow,['BEAR']] = bearing
            xd.loc[irow,['VELO']] = velo
            # other stat output
            xd.loc[irow, ['ESPC']] = VELO.values.std()  # ESPC
            xd.loc[irow,['MAXV']] = VELO.max() # MAXV
            xd.loc[irow,['MINV']] = VELO.min() # MINV
            # (EDVC and ERSC are the same in this subroutine's context)
            xd.loc[irow,['EDVC']] = VELO.size # EDVC Velocity Count
            xd.loc[irow,['ERSC']] = VELO.size # ERSC Spatial Count
                    
        # delete extra lines (nan) not filled above
        xd.dropna(axis=0, how='all', inplace=True)
        # ESPC had several nan's, replace these and other nan before returning?
        xd.fillna(999.000, inplace=True)

        return xd


def qc_radial_file_with_qccodar(radial_file: Radial, qc_values=None, export=None, save_path=None, clean=False, clean_path=None):
    radial_file.threshold_qc_all(qc_values)
    qc_radial_file(radial_file, qc_values=qc_values, export=export, save_path=save_path, clean=clean, clean_path=clean_path)

