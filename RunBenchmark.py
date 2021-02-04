import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import *
import numpy as np
import matplotlib.pyplot as plt
from application import *



#model_name = "fullvar_dw_noisy_v3"
#data_path = "../DataGeneration/TrainingData/testdata_fullvar_dw_noisy.mat"
#inputs, targets = load_thermometry_data(data_path, rescaling_flag=True)
#model = load_model("saved_models/" + model_name + ".json", "saved_models/" + model_name + ".h5")

#dist_out, std_out, temp_bins, temp_mean, counts, sample_sizes = evaluate_model(inputs, targets, model=model, binsize=20, n_repeats=100, plot_results=True, save_report=False)

#np.savetxt('dist.txt', dist_out)  
#np.savetxt('std.txt', std_out)  
#np.savetxt('temps.txt', temp_mean)  
#np.savetxt('sample_sizes.txt', sample_sizes)  
#plt.show()


model_name = "fullvar_dw_noisy_v3"
data_path = '../DataGeneration/TrainingData/testdata_real_N6000s300_T50s5_L100_dw_noisy.mat'

inputs, targets = load_thermometry_data(data_path, rescaling_flag=False)
model = load_model("saved_models/" + model_name + ".json", "saved_models/" + model_name + ".h5")

max_samples = 20
indices = np.arange(inputs.shape[0], dtype=np.int16)
sample_idx = np.random.choice(indices, max_samples, replace=False)

pixelsize = 1.0492e-6
L_cond = 100 # in units of um

for i in range(max_samples):
    profiles = inputs[sample_idx[0:(i+1)],:,:]
    real_temps = targets[sample_idx[0:(i+1)]]
    n1D = pixelsize*np.sum(profiles, axis=1)/L_cond
    n1D = np.expand_dims(n1D, 2)
    profiles_re = profiles*1e-3*n1D**(-3/5)

    T_mean, T_std, mean_pred, std_pred = estimate_temperature(profiles_re, model=model, plot_result=False)


    # plot results
    x_axis = np.arange(1, i+2) 
    plt.figure
    plt.errorbar( x_axis, mean_pred, yerr=std_pred, label='individual predictions')
    plt.plot( x_axis, real_temps, 's', label='actual temperature')
    plt.axhline(T_mean, color='r', label='weighted mean')
    plt.axhline(T_mean+T_std, color='r', linestyle='dashed', label='weighted std')
    plt.axhline(T_mean-T_std, color='r', linestyle='dashed')
    plt.grid(True)
    plt.xlabel('Profile nr.')
    plt.ylabel('Temperature T [nK]')
    plt.legend(loc="upper right")
    plt.title('Temperature estimation') 

    filename = "temp_estimation" + str(i+1) 
    plt.savefig( filename )

    plt.show()


