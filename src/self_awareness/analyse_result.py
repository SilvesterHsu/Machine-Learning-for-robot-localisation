import pickle
import numpy as np

# Display count with leading 0s to make it 3 digits. Easier to sort.
filename = 'results2.pkl'

with open(filename, 'rb') as f:
    results_dict = pickle.load(f)

    trans_errors = results_dict["trans_errors"]
    rot_errors = results_dict["rot_errors"]
    uncertainties = results_dict["uncertainties"]

    num_testing = len(trans_errors)
    intervals = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

    bin_tran_error = []
    bin_rot_error = []
    bin_tran_error2 = []
    bin_rot_error2 = []
    bin_quantities = []

    for i in range(intervals.shape[0]):

        if i == intervals.shape[0]-1:
            upper_value = np.inf
        else:
            upper_value = intervals[i + 1]

        idx = (uncertainties > intervals[i]) * (uncertainties <= upper_value)

        idx = np.nonzero(idx==1)

        tmp1 = trans_errors[idx[0]]
        bin_tran_error.append(np.mean(tmp1))
        bin_tran_error2.append(np.median(tmp1))
        tmp2 = rot_errors[idx[0]]
        bin_rot_error.append(np.mean(tmp2))
        bin_rot_error2.append(np.median(tmp2))

        assert tmp1.shape[0] == tmp2.shape[0]

        bin_quantities.append(tmp1.shape[0]/num_testing)


    print('bin_tran_error: ', bin_tran_error)
    print('bin_rot_error: ', bin_rot_error)
    print('bin_tran_error2: ', bin_tran_error2)
    print('bin_rot_error2: ', bin_rot_error2)
    print('bin_quantities: ', bin_quantities)

    import scipy.io as sio

    sio.savemat('plot.mat', {'bin_tran_error': np.array(bin_tran_error), 'bin_rot_error': np.array(bin_rot_error),
                                'bin_tran_error2': np.array(bin_tran_error2), 'bin_rot_error2': np.array(bin_rot_error2),
                                'bin_quantities': bin_quantities, 'intervals': intervals})