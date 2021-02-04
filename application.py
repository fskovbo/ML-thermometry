import os, json
from keras import *
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def load_thermometry_data(data_path, **opt):
    rescaling_flag = opt.get('rescaling_flag', True)

    # Load data from data_path. Note, should be a .mat file
    data = loadmat(data_path)   

    profiles = data['densities']
    parameters = data['parameters']

    temperatures = parameters[:,0] # in nK
    boxlengths = parameters[:,1] # in um
    atomnumbers = parameters[:,2]


    # Rescale profiles according to mean density
    if rescaling_flag:
        n1D = np.expand_dims( atomnumbers/(boxlengths), 1)
        profiles *= 1e-3 * n1D**(-3/5)

    # Assign inputs and targets for training
    inputs = profiles
    targets = temperatures

    inputs = inputs.astype('float32') 
    targets = targets.astype('float32') 

    if inputs.ndim == 2:
        inputs = np.expand_dims(inputs, 2) # pad with dummy index to work with conv1D

    return inputs, targets


def save_model(model, filename):
    # Save model weights (as .h5) and architecture (as .json)
    h5file = filename + ".h5"
    json_file = filename + ".json"

    model.save_weights(h5file, overwrite=True)
    model_json = model.to_json()
    with open(json_file, "w") as outfile:
        outfile.write(model_json)

    print('files: %s, %s' % (h5file, json_file))


def load_model(model_file, weights_file):
    # Load model from disc
    if not model_file:
        raise Exception("If loading model from disc, please specify filename of model")
    if not weights_file:
        raise Exception("If loading model from disc, please specify filename of weights")
        
    print("loading model from file: %s" % (model_file,))
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    print("loading weights from file: %s" % (weights_file,))
    model.load_weights(weights_file)

    return model


def check_inputs_format(profiles, input_shape):
    # Check length of profiles and add padding if necessary
    N_gridpoints = profiles.shape[1]

    if profiles.ndim == 2:
        profiles = np.expand_dims(profiles, 2) # pad with dummy index to work with conv1D

    if N_gridpoints < input_shape[1]:
        print("Input length of model is %d while profile length is %d. Adding padding to profiles.",input_shape[1], N_gridpoints)
        
        pad_total = input_shape[1] - N_gridpoints

        if pad_total % 2 == 0:
            pad_front = pad_total/2
            pad_back = pad_total/2
        else:
            pad_front = (pad_total+1)/2
            pad_back = (pad_total-1)/2    
        
        npad = ((0, 0), (pad_front, pad_back), (0, 0))
        profiles = np.pad(profiles, pad_width=npad, mode='constant', constant_values=0)

    if N_gridpoints > input_shape[1]:
        raise Exception("Profiles too long! Input length of model is %d while profile length is %d!",input_shape[1], N_gridpoints)

    return profiles



def transform_profiles(profiles, temperatures, **opt):
    # Generate additional training profiles by shifting and/or
    # mirroring the input profiles by a random number of pixels.

    max_shift = opt.get('max_shift', 50) # max number of pixels shifted
    n_shifts = opt.get('n_shifts', 5) # number of times each profile is shifted
    mirror = opt.get('mirror', True) # if true, also mirror each profile


    n_profs = profiles.shape[0]
    profiles_shifted = np.zeros((n_shifts*n_profs, profiles.shape[1],1))
    temps_shifted = np.zeros((n_shifts*n_profs))

    # generate and apply shifts
    shifts = np.random.randint(max_shift+1, size=n_shifts*n_profs)
    directions = np.sign(np.random.normal(size=n_shifts*n_profs))
    shifts = shifts*directions
    indices = np.arange(n_shifts*n_profs).reshape((n_profs,n_shifts))
    for i in range(profiles.shape[0]):
        for j in range(n_shifts):
            ss = int(shifts[indices[i,j]])
            prof_temp = np.roll(profiles[i,:,:], ss, axis=0) # shift spatial axis, here axis=0, since first index is removed upon access

            # set ss first or last entries to zero 
            if ss > 0:
                prof_temp[:ss,0] = 0
            if ss < 0:
                prof_temp[ss:,0] = 0

            profiles_shifted[indices[i,j],:,:] = prof_temp
            temps_shifted[indices[i,j]] = temperatures[i]

    profiles_out = np.concatenate((profiles, profiles_shifted), axis=0)
    temps_out = np.concatenate((temperatures, temps_shifted), axis=0)

    if mirror:
        profiles_mirror = np.flip(profiles_out, axis=1) # flip spatial axis
        profiles_out = np.concatenate((profiles_out, profiles_mirror), axis=0)
        temps_out = np.concatenate((temps_out, temps_out), axis=0)

    return profiles_out, temps_out


def estimate_temperature(profiles, **opt):
    # Use an already trained model to predict the temperatures of 
    # the input profiles.
    # Note, profiles should shape [N_samples, N_gridpoints]

    model = opt.get('model', None)
    model_file = opt.get('model_file', "")
    weights_file = opt.get('weights_file', "")

    use_weighted_mean = opt.get('use_weighted_mean', True)
    plot_result = opt.get('plot_result', True)


    if model is None:
        # No model passed to function. Should be loaded from disc instead.
        model = load_model(model_file, weights_file)

    # Check length of profiles and add padding if necessary
    input_shape = model.layers[0].input_shape
    profiles = check_inputs_format(profiles, input_shape)
    N_samples = profiles.shape[0]

    # Use model to predict temperature and uncertainty
    predictions = model.predict(profiles)

    mean_pred = predictions[:,0]
    std_pred = predictions[:,1]

    if use_weighted_mean:
        # Weigh each point according to the uncertainty of the prediction
        weights = 1/std_pred**2
        weights /= np.sum(weights) # make sure to normalize the weights
    else:
        # Weights are all equal
        weights = np.ones((N_samples,))/N_samples        

    T_mean = np.sum( weights*mean_pred )
    T_std = np.sqrt( np.sum(weights*(mean_pred-T_mean)**2/N_samples) )


    # Plot the results
    if plot_result:
        x_axis = np.arange(1, N_samples+1) 
        plt.figure
        plt.errorbar( x_axis, mean_pred, yerr=std_pred, label='individual predictions')
        plt.axhline(T_mean, color='r', label='weighted mean')
        plt.axhline(T_mean+T_std, color='r', linestyle='dashed', label='weighted std')
        plt.axhline(T_mean-T_std, color='r', linestyle='dashed')
        plt.grid(True)
        plt.xlabel('Profile nr.')
        plt.ylabel('Temperature T [nK]')
        plt.legend(loc="upper right")
        plt.title('Temperature estimation')
        plt.show()

    
    return T_mean, T_std, mean_pred, std_pred


def evaluate_model(inputs, targets, **opt):
    # Bin the targets (temperatures) and evaluate the performance of
    # the model on each bin. Both the performance of predictions on 
    # individual profiles and on batches is checked.

    model = opt.get('model', None)
    model_file = opt.get('model_file', "")
    weights_file = opt.get('weights_file', "")
    data_file = opt.get('data_file', "") 

    binsize = opt.get('binsize', 5)
    sample_sizes = opt.get('sample_sizes', np.array([5, 10, 25, 50], dtype=np.int16) )
    n_repeats = opt.get('n_repeats', 15)
    plot_result = opt.get('plot_result', True)
    save_report = opt.get('save_report', True)
    report_name = opt.get('report_name', "evaluation_report")

    if model is None:
        # No model passed to function. Should be loaded from disc instead.
        model = load_model(model_file, weights_file)

    # Check length of profiles and add padding if necessary
    input_shape = model.layers[0].input_shape
    inputs = check_inputs_format(inputs, input_shape)


    print("\n *** Evaluating performance of model *** \n")

    # Bin the temperatures and get the bin indeces of the targets
    bin_min = np.amin(targets)-binsize/2
    bin_max = np.amax(targets)+binsize/2
    temp_bins = np.arange( bin_min, bin_max, binsize )
    if temp_bins[-1] < np.amax(targets):
        temp_bins = np.append(temp_bins, bin_max)

    n_bins = temp_bins.size - 1

    bin_idx = np.digitize(targets, temp_bins)-1 # must subtract one, as digitize starts indicing from 1
    counts = np.bincount(bin_idx) 


    # Calculate the mean temperature within each bin
    temp_mean = np.zeros((n_bins))
    for i in range(n_bins):
        if counts[i] > 0:
            temp_mean[i] = np.mean( targets[bin_idx == i] )

    
    # Format for output
    sample_sizes_print = ["Single", *sample_sizes]
    row_format ="{:>15}" * 3
    header_format = "\nPerformance on bin no. {:d} with {:d} entries spanning T=[{:.1f}, {:.1f}) nK:\n"

    # For each binned target (temperature) evaluate the performance of the 
    # model. Since the loss function (log-min-likelihood) is a poor measure 
    # of evaluation, instead the mean distance and mean std are examined.
    # For each bin, both the individual performance on all members plus the
    # collective performance on subsets of various sizes are found.
    mean_dist_ind = np.zeros((n_bins,1))
    mean_std_ind = np.zeros((n_bins,1))
    mean_dist_col = np.zeros((n_bins,sample_sizes.size))
    mean_std_col = np.zeros((n_bins,sample_sizes.size))  

    for i in range(n_bins):
        # First, evaluate performance on each profile individually
        if counts[i] == 0:
            mean_dist_ind[i] = np.nan
            mean_std_ind[i] = np.nan           
            mean_dist_col[i,:] = np.nan
            mean_std_col[i,:] = np.nan
            continue

        targets_eval = targets[bin_idx == i]
        inputs_eval = inputs[bin_idx == i, :, :]

        predictions = model.predict(inputs_eval)
        mean_pred = predictions[:,0]
        std_pred = predictions[:,1]

        mean_dist_ind[i] = np.mean(np.sqrt( (mean_pred-targets_eval)**2 ))
        mean_std_ind[i] = np.mean( std_pred )


        # Next, evaluate collective performance on batches
        for j in range(sample_sizes.size):
            
            if counts[i] < sample_sizes[j]:
                mean_dist_col[i,j] = np.nan
                mean_std_col[i,j] = np.nan
                continue

            T_tmp = np.zeros((n_repeats))
            std_tmp = np.zeros((n_repeats))
            for k in range(n_repeats):
                # draw a random subset of bin
                in_bin_filter = bin_idx == i
                in_bin_idx = np.arange(in_bin_filter.size, dtype=np.int16)
                sample_idx = np.random.choice(in_bin_idx[in_bin_filter], sample_sizes[j], replace=False)

                T_tmp[k], std_tmp[k], _, _ = estimate_temperature(inputs[sample_idx,:,:], model=model, plot_result=False)

            mean_dist_col[i,j] = np.mean(np.sqrt( (T_tmp-np.mean(targets[sample_idx]))**2 ))
            mean_std_col[i,j] = np.mean( std_tmp )

        # Display the performance on current bin
        data_print_dist = np.concatenate((mean_dist_ind[i], mean_dist_col[i,:]))
        data_print_std = np.concatenate((mean_std_ind[i], mean_std_col[i,:]))

        print(header_format.format(i+1, counts[i], temp_bins[i], temp_bins[i+1]))
        print(row_format.format("Sample size", "Mean dist.", "Mean std"))

        for ss, dist, std in zip(sample_sizes_print, data_print_dist, data_print_std):
            dist = np.around(dist, decimals=1)
            std = np.around(std, decimals=1)
            print(row_format.format(ss, dist, std))


    dist_out = np.concatenate((mean_dist_ind, mean_dist_col), axis=1)
    std_out = np.concatenate((mean_std_ind, mean_std_col), axis=1)


    # Plot the results
    if plot_result:
        plt.figure 
        plt.subplot(sample_sizes.size+1, 2, 1)
        plt.plot(temp_mean, mean_dist_ind, label="Single")
        plt.grid(True)
        plt.xlabel('Mean temperature in bin [nK]')
        plt.ylabel('dist.')
        plt.legend(loc="upper right")

        plt.subplot(sample_sizes.size+1, 2, 2)
        plt.plot(temp_mean, mean_std_ind[:,0], label="Single")
        plt.grid(True)
        plt.xlabel('Mean temperature in bin [nK]')
        plt.ylabel('std.')
        plt.legend(loc="upper right")

        for j in range(sample_sizes.size):
            plt.subplot(sample_sizes.size+1, 2, 2*j+3)
            plt.plot(temp_mean, mean_dist_col[:,j], label="Sample size = " + str(sample_sizes[j]))
            plt.grid(True)
            plt.xlabel('Mean temperature in bin [nK]')
            plt.ylabel('dist.')
            plt.legend(loc="upper right")

            plt.subplot(sample_sizes.size+1, 2, 2*j+4)
            plt.plot(temp_mean, mean_std_col[:,j], label="Sample size = " + str(sample_sizes[j]))
            plt.grid(True)
            plt.xlabel('Mean temperature in bin [nK]')
            plt.ylabel('std.')
            plt.legend(loc="upper right")


    # Save report on the benchmark
    if save_report:
        # Save report in format readable for humans
        with open( report_name + ".txt", "w") as text_file:
            print(" *** EVALUATION SETTINGS *** \n", file=text_file)
            print("Model filename: {}".format(model_file), file=text_file)
            print("Weights filename: {}".format(weights_file), file=text_file)
            print("Evaluation data filename: {}\n".format(data_file), file=text_file)
            print("Binsize: {:.2f}".format(binsize), file=text_file)
            print("Sample sizes: {}".format(sample_sizes), file=text_file)
            print("Number of repeat samples: {:d}".format(n_repeats), file=text_file)
            print("Number of bins: {:d}\n\n\n".format(n_bins), file=text_file)
            
            print(" *** EVALUATION RESULTS *** ", file=text_file)
            for i in range(n_bins):
                print(header_format.format(i+1, counts[i], temp_bins[i], temp_bins[i+1]), file=text_file)

                if counts[i] == 0:
                    continue

                print(row_format.format("Sample size", "Mean dist.", "Mean std"), file=text_file)

                for ss, dist, std in zip(sample_sizes_print, dist_out[i,:], std_out[i,:]):
                    dist = np.around(dist, decimals=1)
                    std = np.around(std, decimals=1)
                    print(row_format.format(ss, dist, std), file=text_file)

    return dist_out, std_out, temp_bins, temp_mean, counts, sample_sizes


def save_training_history(history, filename):
    # Save training history as a .json file such that it can 
    # be plotted later. Also write the training history to a 
    # readable text file. 

    # with open(filename + ".json", "w") as json_file:
    #     json.dump(history, json_file)

    n_epochs = len(history['loss'])
    n_cols = (len(history) + 1)
    row_format ="{:>20}" * n_cols


    with open(filename + ".txt", "w") as text_file:
        print(row_format.format(*['epoch', *[*history]]), file=text_file)

        for i in range(n_epochs):
            vals = np.zeros((n_cols))
            vals[0] = i+1
            idx = 1
            for key in history:
                vals[idx] = "%.4f" % history[key][i]
                idx += 1

            print(row_format.format(*vals), file=text_file)