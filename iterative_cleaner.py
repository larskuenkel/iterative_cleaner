#!/usr/bin/env python

# Tool to remove RFI from pulsar archives.
# Originally written by Patrick Lazarus. Modified by Lars Kuenkel.

from __future__ import print_function
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.optimize
import argparse
import psrchive


def parse_arguments():
    parser = argparse.ArgumentParser(description='Commands for the cleaner')
    parser.add_argument('archive', nargs='+', help='The chosen archives')
    parser.add_argument('-c', '--chanthresh', type=float, default=5, metavar=('channel_threshold'), help='The threshold (in number of sigmas) a '
                                                                    'profile needs to stand out compared to '
                                                                    'others in the same channel for it to '
                                                                    'be removed.')
    parser.add_argument('-s', '--subintthresh', type=float, default=5, metavar=('subint_threshold'), help='The threshold (in number of sigmas) a '
                                                                    'profile needs to stand out compared to '
                                                                    'others in the same subint for it to '
                                                                    'be removed.')
    parser.add_argument('-m', '--max_iter', type=int, default=5, metavar=('maximum_iterations'), help='Maximum number of iterations.')
    parser.add_argument('-z', '--print_zap', action='store_true', help='Creates a plot that shows which profiles get zapped.')
    parser.add_argument('-u', '--unload_res', action='store_true', help='Creates an archive that contains the pulse free residual.')
    parser.add_argument('-p', '--pscrunch', action='store_true', help='Pscrunches the output archive.')
    parser.add_argument('-q', '--quiet', action='store_true', help='Do not print cleaning information.')
    parser.add_argument('-l', '--no_log', action='store_true', help='Do not create cleaning log.')
    parser.add_argument('-r', '--pulse_region', nargs=3, type=float, default=[0,0,1], 
        metavar=('pulse_start', 'pulse_end', 'scaling_factor'), help="Defines the range of the pulse and a suppression factor.")
    parser.add_argument('-o', '--output', type=str, default='', metavar=('output_filename'), 
        help="Name of the output file. If set to 'std' the pattern NAME.FREQ.MJD.ar will be used.")
    parser.add_argument('--memory', action='store_true', help='Do not pscrunch the archive while it is in memory.\
                                                                Costs RAM but prevents having to reload the archive.')
    parser.add_argument('--bad_chan', type=float, default=1, help='Fraction of subints that needs to be removed in order to remove the whole channel.')
    parser.add_argument('--bad_subint', type=float, default=1, help='Fraction of channels that needs to be removed in order to remove the whole subint.')
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
        if not args.quiet:
            print("Cleaned archive: %s" % o_name)


def clean(ar, args, arch):
    orig_weights = ar.get_weights()
    if args.memory and not args.pscrunch:
        pass
    else:
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
    if not args.quiet:
        print(("Total number of profiles: %s" % profile_number))
    while x < max_iterations:
        x += 1
        if not args.quiet:
            print(("Loop: %s" % x))

        # Prepare the data for template creation
        patient.pscrunch()  # pscrunching again is not necessary if already pscrunched but prevents a bug
        patient.remove_baseline()
        patient.dedisperse()
        patient.fscrunch()
        patient.tscrunch()
        template = patient.get_Profile(0, 0, 0).get_amps() * 10000

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
        rfi_frac = (new_weights.size - np.count_nonzero(new_weights)) / float(new_weights.size)

        # Print the changes to the previous loop to help in choosing a suitable max_iter
        if not args.quiet:
            print(("Differences to previous weights: %s  RFI fraction: %s" %(diff_weigths, rfi_frac)))
        for old_weights in test_weights:
            if np.all(new_weights == old_weights):
                if not args.quiet:
                    print(("RFI removal stops after %s loops." % x))
                loops = x
                x = 1000000
        test_weights.append(new_weights)

    if x == max_iterations:
        if not args.quiet:
            print(("Cleaning was interrupted after the maximum amount of loops (%s)" % max_iterations))
        loops = max_iterations

    # Reload archive if it is not supposed to be pscrunched.
    if not args.pscrunch and not args.memory:
        ar = psrchive.Archive_load(arch)

    # Set weights in archive.
    set_weights_archive(ar, avg_test_results)

    # Test if whole channel or subints should be removed
    if args.bad_chan != 1 or args.bad_subint != 1:
        ar = find_bad_parts(ar, args)


    # Unload residual if needed
    if args.unload_res:
        residual.unload("%s_residual_%s.ar" % (ar_name, loops))

    # Create plot that shows zapped( red) and unzapped( blue) profiles if needed
    if args.print_zap:
        plt.imshow(avg_test_results.T, vmin=0.999, vmax=1.001, aspect='auto',
                interpolation='nearest', cmap=cm.coolwarm)
        plt.gca().invert_yaxis()
        plt.title("%s cthresh=%s sthresh=%s" % (ar_name, args.chanthresh, args.subintthresh))
        plt.savefig("%s_%s_%s.png" % (ar_name, args.chanthresh,
            args.subintthresh), bbox_inches='tight')

    # Create log that contains the used parameters
    if not args.no_log:
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
            args: argparse namepsace object that need to contain the
                following two parameters:
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
            channel = array2d[:, ichan]
            median = np.ma.median(channel)
            channel_rescaled = channel - median
            mad = np.ma.median(np.abs(channel_rescaled))
            scaled[:, ichan] = (channel_rescaled) / mad
    return scaled


def subint_scaler(array2d):
    """For each sub-int scale it.
    """
    scaled = np.empty_like(array2d)
    nsubs = array2d.shape[0]
    for isub in np.arange(nsubs):
        with np.errstate(invalid='ignore', divide='ignore'):
            subint = array2d[isub, :]
            median = np.ma.median(subint)
            subint_rescaled = subint - median
            mad = np.ma.median(np.abs(subint_rescaled))
            scaled[isub, :] = (subint_rescaled) / mad
    return scaled


def remove_profile_inplace(ar, template, pulse_region):
    """Remove the temnplate pulse from the individual profiles.
    """
    data = ar.get_data()[:, 0, :, :]  # Select first polarization channel
                                # archive is P-scrunched, so this is
                                # total intensity, the only polarization
                                # channel
    for isub, ichan in np.ndindex(ar.get_nsubint(), ar.get_nchan()):
        amps = remove_profile1d(data[isub, ichan], isub, ichan, template, pulse_region)[1]
        prof = ar.get_Profile(isub, 0, ichan)
        if amps is None:
            prof.set_weight(0)
        else:
            prof.get_amps()[:] = amps


def remove_profile1d(prof, isub, ichan, template, pulse_region):

    err = lambda amp: amp * template - prof
    params, status = scipy.optimize.leastsq(err, [1.0])
    err2 = np.asarray(err(params))
    if pulse_region != [0, 0, 1]:
        p_start = int(pulse_region[1])
        p_end = int(pulse_region[2])
        err2[p_start:p_end] = err2[p_start:p_end] * pulse_region[0]
    if status not in (1, 2, 3, 4):
        print("Bad status for least squares fit when removing profile.")
        return (isub, ichan), np.zeros_like(prof)
    else:
        return (isub, ichan), err2


def apply_weights(data, weights):
    """Apply the weigths to an array.
    """
    nsubs, nchans, nbins = data.shape
    for isub in range(nsubs):
        data[isub] = data[isub] * weights[isub, ..., np.newaxis]
    return data


def set_weights_archive(archive, test_results):
    """Apply the weigths to an archive according to the test results.
    """
    for (isub, ichan) in np.argwhere(test_results >= 1):
        integ = archive.get_Integration(int(isub))
        integ.set_weight(int(ichan), 0.0)


def find_bad_parts(archive, args):
    """Checks whether whole channels or subints should be removed
    """
    weights = archive.get_weights()
    n_subints = archive.get_nsubint()
    n_channels = archive.get_nchan()
    n_bad_channels = 0
    n_bad_subints = 0

    for i in range(n_subints):
        bad_frac = 1 - np.count_nonzero(weights[i, :]) / float(n_channels)
        if bad_frac > args.bad_subint:
            for j in range(n_channels):
                integ = archive.get_Integration(int(i))
                integ.set_weight(int(j), 0.0)
            n_bad_subints += 1

    for j in range(n_channels):
        bad_frac = 1 - np.count_nonzero(weights[:, j]) / float(n_subints)
        if bad_frac > args.bad_chan:
            for i in range(n_subints):
                integ = archive.get_Integration(int(i))
                integ.set_weight(int(j), 0.0)
            n_bad_channels += 1

    if not args.quiet and n_bad_channels + n_bad_subints != 0:
        print(("Removed %s bad subintegrations and %s bad channels." % (n_bad_subints, n_bad_channels)))
    return archive


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
