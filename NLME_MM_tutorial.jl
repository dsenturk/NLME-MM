###########################################################################################
## Description: A step-by-step implementation of MM algorithm for nonlinear mixed-effects 
##              models (NLME), and associated procedures including simulated data 
##              generation and model fitting described in "Modeling intra-individual 
##              inter-trial EEG response variability in Autism Spectrum Disorder".
###########################################################################################
## Main functions implemented:
##    1. NLME_single_generate: Function that generates trial-level ERP responses for fitting
##                             single-level NLME model
##    2. fit_nlme_mm!: Function that fits single-level NLME model via MM algorithm
##    3. NLME_multi_generate: Function that generates trial-level ERP responses for fitting
##                            multi-level NLME model
##    4. fit_mnlme_mm!: Function that fits multi-level NLME model via MM algorithm
###########################################################################################
## Required files:
##    1. preparation_functions.jl
##    2. NLME_classfiles.jl
##    3. NLME_MM_single.jl
##    4. NLME_single_simulation.jl
##    5. NLME_MM_multi.jl
##    6. NLME_multi_simulation.jl
###########################################################################################


###########################################################################################
# Ensure necessary packages and install missing ones
###########################################################################################
# using Pkg
# installed_packages = keys(Pkg.dependencies())
# packages_to_ensure = ["Distributed", "CSV", "DataFrames", "Distributions", "Plots",
#                       "LinearAlgebra", "SparseArrays", "JLD2", "SharedArrays"]
# for pkg_name in packages_to_ensure
#     if !(pkg_name in installed_packages)
#         Pkg.add(pkg_name)
#     end
# end


###########################################################################################
# Load packages and required files
###########################################################################################
using Distributed
using CSV
using DataFrames
using Plots
addprocs(4) # number of parallel computation nodes you want to use
@everywhere begin
    using Distributions
    using LinearAlgebra
    using SparseArrays
    using JLD2
    using SharedArrays
    include("preparation_functions.jl")
    include("NLME_classfiles.jl")
    include("NLME_MM_single.jl")
    include("NLME_MM_multi.jl")
    include("NLME_single_simulation.jl")
    include("NLME_multi_simulation.jl")
end

###########################################################################################
# 1. One run of single-level NLME simulation described in Section 4.1
###########################################################################################

# 1.1 generate a 200-trial simulated trial-level EEG data set for single-level NLME ######
single_dt = NLME_single_generate(200,                               # R: number of trials
                                [6.0, 5.0, 6.0, 14.0, -5.0],        # beta_true: true fixed effects
                                Diagonal([1.5,4.0,2.0,6.0,10.0]),   # V_true: true covariance matrix of trial-level random effects
                                8.0,                                # sigma2_ture: true variance of measurement error
                                collect(0:0.8:20))                  # t0: trial time grid

# 1.2 calculate starting values of model components ########################################
start_nlme!(single_dt,                    # m: `NlmeModel` object containing data and model components
                                          #    (list of values see in "NLME_classfile.jl)
            [13.0,17.0],                  # peak_range: searching interval of peak-shaped component
            [3.0,7.0])                    # dip_range: searching interval of dip-shaped component


# 1.3 fit single-level NLME model using MM algorithm #######################################
single_mod = fit_nlme_mm!(single_dt,    # m: `NlmeModel` object containing data and model components
                                        #    (list of values see in "NLME_classfile.jl)
                           20,          # gn_maxiter: maximum number of iterations of Gauss-Newton algorithm in Step 1         
                           1e-4,        # gn_err_threshold: relative convergence threshold for Gauss-Newton algorithm
                           20,          # halving_maxiter: maximum times of step-halving
                           20,          # maxiter: maximum number of iterations for MM algorithm
                           1e-4)        # logl_err_threshold: relative convergence threshold for MM algorithm

# 1.4 print out estimated model components #################################################
println(string("estimated fixed effects: beta = ", single_mod.beta), "\n")
println(string("estimated trial-level variance components: V = diag(", diag(single_mod.V) ,")\n"))
println(string("estimated variance of measurement error: sigma2 = ", single_mod.sigma2), "\n")

# 1.5 plot raw data and prediction for a single trial #######################################
y_pred = Matrix{Float64}(undef,
                size(single_dt.t,1), 
                single_dt.R)
for r in 1:single_dt.R
    y_pred[:,r] = mu(single_dt.t, single_dt.beta + single_dt.gamma[:,r])
end

trial = 1:3       # single (or multiple) trial(s) for plotting (any number between 1 and R)
plot(single_dt.t, single_dt.y_mt[:,trial], color="black", label = "")       # raw data in black
plot!(single_dt.t, y_pred[:,trial], color="red", label = "")                # prediction in red
plot!([NaN, NaN],[NaN, NaN], color = "black", label = "raw data")
plot!([NaN, NaN],[NaN, NaN], color = "red", label = "prediction")





###########################################################################################
# 2. One run of multi-level NLME simulation described in Section 4.2
###########################################################################################


# 2.1 generate a 50-subject 50-trial-per-subject simulated data set for multi-level NLME ###
multi_dt = NLME_multi_generate(50,                                  # n: number of subjects
                               50,                                  # R: number of trials per subject
                               [6.0, 5.0, 6.0, 14.0, -5.0],         # beta_true: true fixed effects
                               Diagonal([1.0,3.0,1.0,3.0,10.0]),    # U_true: true covariance matrix of subject-level random effects
                               Diagonal([1.0,5.0,1.5,6.0,15.0]),    # V_true: true covariance matrix of trial-level random effects
                               8.0,                                 # sigma2_true: true variance of measurement error
                               collect(0:0.8:20))                   # t0: trial time grid


# 2.2 calculate starting values of model components ########################################
start_mnlme!(multi_dt,          # m: 'MnlmeModel' object containing data and model components
                                #    (list of values seee in "NLME_classfile.jl")
             [13.0, 17.0],      # peak_range: searching interval of peak-shpaed component
             [3.0, 7.0],        # dip_range: seraching interval of dip-shaped component
             10,                # gn_maxiter: maximum number of iterations for Gauss-Newton algorithm in single-level NLME
             1e-4,              # gn_err_threshold: relative convergence threshold for Gauss-Newton algorithm in single-level NLME
             20,                # halving_maxiter: maximum times of step-halving in single-level NLME
             20,                # maxiter: maximum number of iterations for MM algorithm for single-level NLME
             1e-4)              # logl_err_threshold: relative convergence threshold for MM algorithm for single-level NLME


# 2.3 fit single-level NLME model using MM algorithm #######################################
multi_mod = fit_mnlme_mm!(multi_dt, # m: 'MnlmeModel' object containing data and model components
                                    #    (list of values seee in "NLME_classfile.jl") 
                          10,       # gn_maxiter: maximum number of iterations for Gauss_Newton algorithm in multi-level NLME
                          1e-4,     # gn_err_threshold: relative convergence threshold for Gauss-Newton algorithm in multi-level NLME
                          20,       # halving_maxiter: maximum times of step-halving in multi-level NLME
                          20,       # maxiter: maximum number of iterations for MM algorithm for multi-level NLME
                          1e-4)     # logl_err_threshold: relative convergence threshold for MM algorithm for multi-level NLME


# 2.4 print out estimated model components #################################################
println(string("estimated fixed effects: beta = ", multi_mod.beta), "\n")
println(string("estimated subject-level variance components: U = diag(", diag(multi_mod.U) ,")\n"))
println(string("estimated trial-level variance components: V = diag(", diag(multi_mod.V) ,")\n"))
println(string("estimated variance of measurement error: sigma2 = ", multi_mod.sigma2), "\n")

# 2.5 plot raw data and prediction for a single trials #######################################
y_pred_array = Array{Matrix{Float64}}(undef, multi_dt.n)    # generate array of matrices of length n

for i in 1:multi_dt.n
    # set dimension of prediction matrix (T: number of trial time point; R_i: number of trials of i-th subject)
    y_pred_array[i] = zeros(Float64, multi_dt.T, multi_dt.R_array[i])
    phi_i = multi_mod.beta + multi_mod.alpha[:,i]
    for r in 1:multi_dt.R_array[i]
        y_pred_array[i][:,r] = mu(multi_dt.t, phi_i + multi_dt.gamma[i][:,r])
    end
end

subject = 1       # one subject for plotting (i-th subject)
trial = 1:3       # single(or multiple) trial(s) for plotting (any number between 1 and R_i)
plot(multi_dt.t, multi_dt.y_mt_array[subject][:,trial], color="black", label = "")       # raw data in black
plot!(multi_dt.t, y_pred_array[subject][:,trial], color="red", label = "")               # prediction in red
plot!([NaN, NaN],[NaN, NaN], color = "black", label = "raw data")
plot!([NaN, NaN],[NaN, NaN], color = "red", label = "prediction")
