# Originally written by Patrick Lazarus. Modified by Lars Kuenkel.

import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.optimize
import argparse
import psrchive
import multiprocessing
# from memory_profiler import profile
# import time



def parse_arguments():
    parser = argparse.ArgumentParser(description='Commands for the cleaner')
    parser.add_argument('archive', nargs='+', help='The chosen archives')
    parser.add_argument('-c', '--chanthresh', type=float, default=5, metavar=('channel_threshold'), help='The threshold (in number of sigmas) a ' \
                                                                    'profile needs to stand out compared to ' \
                                                                    'others in the same channel for it to ' \
                                                                    'be removed.')
    parser.add_argument('-s', '--subintthresh', type=float, default=5, metavar=('subint_threshold'), help='The threshold (in number of sigmas) a ' \
                                                                    'profile needs to stand out compared to ' \
                                                                    'others in the same subint for it to ' \
                                                                    'be removed.')
    parser.add_argument('-m', '--max_iter', type=int, default=5, metavar=('maximum_iterations'), help='Maximum number of iterations.')
    parser.add_argument('-z', '--print_zap', action='store_true', help='Creates a plot that shows which profiles get zapped.')
    parser.add_argument('-u', '--unload_res', action='store_true', help='Creates an archive that contains the pulse free residual.')
    parser.add_argument('-p', '--pscrunch', action='store_true', help='Pscrunches the output archive.')
    parser.add_argument('-r', '--pulse_region', nargs=3, type=float, default=[0,0,1], 
        metavar=('pulse_start', 'pulse_end', 'scaling_factor'), help="Defines the range of the pulse and a suppression factor.")
    parser.add_argument('-o', '--output', type=str, default='', metavar=('output_filename'), 
        help="Name of the output file. If set to 'std' the pattern NAME.FREQ.MJD.ar will be used.")
    args = parser.parse_args()
    return args


def main(args):
    for arch in args.archive:
        ar = psrchive.Archive_load(arch)
        if args.output == '':
            orig_name = str(ar).split(':', 1)[1].strip()
            o_name = orig_name + '_cleaned.ar'
        else:
            if args.output == 'std':
                mjd = (float(ar.start_time().strtempo()) + float(ar.end_time().strtempo())) / 2.0
                name = ar.get_source()
                cent_freq = ar.get_centre_frequency()
                o_name = "%s.%.3f.%f.ar" % (name, cent_freq, mjd)
            else:
                o_name = args.output
        ar = clean(ar, args, arch)
        ar.unload(o_name)
        print "Cleaned archive: %s" % o_name


# @profile
def clean(ar, args, arch):
    orig_weights = ar.get_weights()
    ar.pscrunch()
    patient = ar.clone()
    ar_name = ar.get_filename().split()[-1]
    x = 0
    max_iterations = args.max_iter
    pulse_region = args.pulse_region

    # Create list that is used to end the iteration
    test_weights = []
    test_weights.append(patient.get_weights())
    profile_number = orig_weights.size
    print ("Total number of profiles: %s" % profile_number)
    while x < max_iterations:
        x += 1
        print ("Loop: %s" % x)

        # Prepare the data for template creation
        patient.pscrunch()
        patient.remove_baseline()
        patient.dedisperse()
        patient.fscrunch()
        patient.tscrunch()
        template = patient.get_Profile(0, 0, 0).get_amps()*10000

        # Reset patient
        patient = ar.clone()
        patient.pscrunch()
        patient.remove_baseline()
        patient.dedisperse()
        remove_profile_inplace(patient, template, pulse_region)

        # re-set DM to 0
        patient.dededisperse()
        
        if args.unload_res:

            residual = patient.clone()

        # Get data (select first polarization - recall we already P-scrunched)
        data = patient.get_data()[:, 0, :, :]
        data = apply_weights(data, orig_weights)

        # Mask profiles where weight is 0
        mask_2d = np.bitwise_not(np.expand_dims(orig_weights, 2).astype(bool))
        mask_3d = mask_2d.repeat(ar.get_nbin(), axis=2)
        data = np.ma.masked_array(data, mask=mask_3d)

        # RFI-ectomy must be recommended by average of tests
        avg_test_results = comprehensive_stats(data, args, axis=2)

        # Reset patient and set weights in patient
        del patient
        patient = ar.clone()
        set_weights_archive(patient, avg_test_results)

        # Test whether weigths were already used in a previous iteration
        new_weights = patient.get_weights()
        diff_weigths = np.sum(new_weights != test_weights[-1])
        rfi_frac = (new_weights.size - np.sum(new_weights)) / new_weights.size

        # Print the changes to the previous loop to help in choosing a suitable max_iter
        print ("Differences to previous weights: %s  RFI fraction: %s" %(diff_weigths, rfi_frac))
        for old_weights in test_weights:
            if np.all(new_weights == old_weights):
                print ("RFI removal stops after %s loops." % x)
                loops = x
                x = 1000000
        test_weights.append(new_weights)

    if x == max_iterations:
        print ("Cleaning was interrupted after the maximum amount of loops (%s)" % max_iterations)
        loops = max_iterations

    # Reload archive if it is not supposed to be pscrunched.
    if not args.pscrunch:
        ar = psrchive.Archive_load(arch)
    
    # Set weights in archive.
    set_weights_archive(ar, avg_test_results)

    # Unload residual if needed
    if args.unload_res:
        residual.unload("%s_residual_%s.ar" % (ar_name, loops))

    # Create plot that shows zapped( red) and unzapped( blue) profiles if needed
    if args.print_zap:
        plt.imshow(avg_test_results.T, vmin=0.999, vmax=1.001, aspect='auto'
            , interpolation='nearest', cmap=cm.coolwarm)
        plt.gca().invert_yaxis()
        plt.title("%s cthresh=%s sthresh=%s" % (ar_name, args.chanthresh, args.subintthresh))
        plt.savefig("%s_%s_%s.png" % (ar_name, args.chanthresh, 
            args.subintthresh), bbox_inches='tight')

    # Create log that contains the used parameters
    with open("clean.log", "a") as myfile:
        myfile.write("\n %s: Cleaned %s with %s, required loops=%s"
         % (datetime.datetime.now(), ar_name, args, loops))
    return ar



def comprehensive_stats(data, args, axis):
    """The comprehensive scaled stats that are used for
        the "Surgical Scrub" cleaning strategy.

        Inputs:
            data: A 3-D numpy array.
            axis: The axis that should be used for computing stats.
            chanthresh: The threshold (in number of sigmas) a
                profile needs to stand out compared to others in the
                same channel for it to be removed.
                (Default: use value defined in config files)
            subintthresh: The threshold (in number of sigmas) a profile
                needs to stand out compared to others in the same
                sub-int for it to be removed.
                (Default: use value defined in config files)

        Output:
            stats: A 2-D numpy array of stats.
    """
    chanthresh = args.chanthresh
    subintthresh = args.subintthresh

    nsubs, nchans, nbins = data.shape
    diagnostic_functions = [
        np.ma.std,
        np.ma.mean,
        np.ma.ptp,
        lambda data, axis: np.max(np.abs(np.fft.rfft(
            data - np.expand_dims(data.mean(axis=axis), axis=axis),
            axis=axis)), axis=axis)
    ]
    # Compute diagnostics
    diagnostics = []
    for func in diagnostic_functions:
        diagnostics.append(func(data, axis=2))

    # Now step through data and identify bad profiles
    scaled_diagnostics = []
    for diag in diagnostics:
        chan_scaled = np.abs(channel_scaler(diag)) / chanthresh
        subint_scaled = np.abs(subint_scaler(diag)) / subintthresh
        scaled_diagnostics.append(np.max((chan_scaled, subint_scaled), axis=0))
    test_results = np.median(scaled_diagnostics, axis=0)
    return test_results


def channel_scaler(array2d):
    """For each channel scale it.
    """
    scaled = np.empty_like(array2d)
    nchans = array2d.shape[1]
    for ichan in np.arange(nchans):
        with np.errstate(invalid='ignore', divide='ignore'):
            detrended = array2d[:, ichan]
            median = np.ma.median(detrended)
            mad = np.ma.median(np.abs(detrended - median))
            scaled[:, ichan] = (detrended - median) / mad
    return scaled


def subint_scaler(array2d):
    """For each sub-int scale it.
    """
    scaled = np.empty_like(array2d)
    nsubs = array2d.shape[0]
    for isub in np.arange(nsubs):
        with np.errstate(invalid='ignore', divide='ignore'):
            detrended = array2d[isub, :]
            median = np.ma.median(detrended)
            mad = np.ma.median(np.abs(detrended - median))
            scaled[isub, :] = (detrended - median) / mad
    return scaled


def remove_profile_inplace(ar, template, pulse_region, nthreads=1):
    data = ar.get_data()[:,0,:,:] # Select first polarization channel
                                  # archive is P-scrunched, so this is
                                  # total intensity, the only polarization
                                  # channel
    if nthreads == 1:
        for isub, ichan in np.ndindex(ar.get_nsubint(), ar.get_nchan()):
            amps = remove_profile1d(data[isub, ichan], isub, ichan, template, pulse_region)[1]
            prof = ar.get_Profile(isub, 0, ichan)
            if amps is None:
                prof.set_weight(0)
            else:
                prof.get_amps()[:] = amps
    else:
        pool = multiprocessing.Pool(processes=nthreads)
        results = []
        for isub, ichan in np.ndindex(ar.get_nsubint(), ar.get_nchan()):
            results.append(pool.apply_async(remove_profile1d, \
                            args=(data[isub, ichan], isub, ichan, template, pulse_region)))
        pool.close()
        pool.join()
        for result in results:
            result.successful()
            (isub, ichan), amps = result.get()
            prof = ar.get_Profile(isub, 0, ichan)
            if amps is None:
                prof.set_weight(0)
            else:
                prof.get_amps()[:] = amps


def remove_profile1d(prof, isub, ichan, template, pulse_region):

    err = lambda amp: amp*template - prof
    params, status = scipy.optimize.leastsq(err, [1.0])
    err2 = np.asarray(err(params))
    if pulse_region != [0, 0, 1]:
        p_start = int(pulse_region[1])
        p_end = int(pulse_region[2])
        err2[p_start:p_end] = err2[p_start:p_end] * pulse_region[0]
    if status not in (1,2,3,4):
        print "Bad status for least squares fit when " \
                            "removing profile."
        return (isub, ichan), np.zeros_like(prof)
    else:
        return (isub, ichan), err2

def apply_weights(data, weights):
    """Apply the weigths to an array.
    """	
    nsubs, nchans, nbins = data.shape
    for isub in range(nsubs):
        data[isub] = data[isub]*weights[isub,...,np.newaxis]
    return data

def set_weights_archive(archive, test_results):
    """Apply the weigths to an archive according to the test results.
    """
    for (isub, ichan) in np.argwhere(test_results >= 1):
        integ = archive.get_Integration(int(isub))
        integ.set_weight(int(ichan), 0.0)


if __name__=="__main__":
    args = parse_arguments()
    main(args)
