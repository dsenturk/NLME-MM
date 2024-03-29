###########################################################################################
## Description: A step-by-step implementation of MM algorithm for nonlinear mixed-effects 
##              models (NLME), and associated procedures including simulated data 
##              generation and model fitting described in "Modeling intra-individual 
##              inter-trial EEG response variability in Autism Spectrum Disorder".
###########################################################################################
## Main functions implemented:
## 1. NLMEs with independent random effects
##    1.1 nlme_indep_generate: Function that generate trial-level ERP responses to fit
##                             single-level NLME model with independent random effects
##    1.2 fit_nlme_indep_mm!: Function that fits single-level NLME model with independent 
##                            random effects via MM algorithm
##    1.3 mnlme_indep_generate: Function that generates trial-level ERP responses to fit
##                              multi-level NLME model with independent random effects
##    1.4 fit_mnlme_indep_mm!: Function that fits multi-level NLME model with independent
##                             randome ffects via MM algorithm
## 2. NLMEs with unstructured random effects
##    2.1 nlme_unstr_generate: Function that generate trial-level ERP responses to fit
##                             single-level NLME model with unstructured random effects
##    2.2 fit_nlme_unstr_mm!: Function that fits single-level NLME model with unstructured 
##                            random effects via MM algorithm
##    2.3 mnlme_unstr_generate: Function that generates trial-level ERP responses to fit
##                              multi-level NLME model with unstructured random effects
##    2.4 fit_mnlme_unstr_mm!: Function that fits multi-level NLME model with unstructured
##                             randome ffects via MM algorithm
###########################################################################################
## Required files:
##    1. preparation_functions.jl
##    2. nlme_classfiles.jl
##    3. nlme_mm.jl
##    4. mnlme_mm.jl
##    5. simulation_generate.jl
###########################################################################################




###########################################################################################
# Ensure necessary packages and install missing ones
###########################################################################################
# using Pkg
# installed_packages = keys(Pkg.dependencies())
# packages_to_ensure = ["Distributed", "CSV", "DataFrames", "Distributions", "Plots", 
#                       "LaTeXStrings", "LinearAlgebra", "SparseArrays", "JLD2", 
#                       "SharedArrays"]
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
using LaTeXStrings
addprocs(4) # number of parallel computation nodes you want to use
@everywhere begin
    using Distributions
    using LinearAlgebra
    using SparseArrays
    using JLD2
    using SharedArrays
    include("preparation_functions.jl")
    include("nlme_classfiles.jl")
    include("nlme_mm.jl")
    include("mnlme_mm.jl")
    include("simulation_generate.jl")
end



###########################################################################################
# 1. Single-level NLME simulation described in Section 4.1
###########################################################################################
## 1.1 One run of single-level NLME simulation with independent random effects
###########################################################################################

# 1.1.1 generate a 200-trial simulated trial-level EEG data set for single-level NLME ######
single_indep_dt = nlme_indep_generate(200,                               # R: number of trials
                                     [6.0, 5.0, 6.0, 14.0, -5.0],        # beta_true: true fixed effects
                                     Diagonal([1.5,4.0,2.0,6.0,10.0]),   # V_true: true covariance matrix of trial-level random effects
                                     8.0,                                # sigma2_ture: true variance of measurement error
                                     collect(0:0.8:20))                  # t0: trial time grid

# 1.1.2 calculate starting values of model components ########################################
start_nlme_indep!(singl_indep_dt,           # m: `NlmeModel` object containing data and model components
                                            #    (list of values see in "nlme_classfiles.jl)
                    [13.0,17.0],            # peak_range: searching interval of peak-shaped component
                    [3.0,7.0])              # dip_range: searching interval of dip-shaped component


# 1.1.3 fit single-level NLME model using MM algorithm #######################################
single_indep_mod = fit_nlme_mm_indep!(single_indep_dt,      # m: `NlmeModel` object containing data and model components
                                                            #    (list of values see in "nlme_classfiles.jl)
                                        20,          # gn_maxiter: maximum number of iterations of Gauss-Newton algorithm in Step 1         
                                        1e-4,        # gn_err_threshold: relative convergence threshold for Gauss-Newton algorithm
                                        20,          # halving_maxiter: maximum times of step-halving
                                        20,          # maxiter: maximum number of iterations for MM algorithm
                                        1e-4)        # logl_err_threshold: relative convergence threshold for MM algorithm

# 1.1.4 print out estimated model components #################################################
println(string("estimated fixed effects (independent): beta = ", singl_indep_dt.beta), "\n")
println(string("estimated trial-level variance components (independent): V = diag(", diag(singl_indep_dt.V) ,")\n"))
println(string("estimated variance of measurement error (independent): sigma2 = ", singl_indep_dt.sigma2), "\n")

# 1.1.5 plot raw data and prediction for a single trial #######################################
y_pred = Matrix{Float64}(undef,
                size(single_indep_dt.t,1), 
                single_indep_dt.R)
for r in 1:single_indep_dt.R
    y_pred[:,r] = mu(single_indep_dt.t, single_indep_dt.beta + single_indep_dt.gamma[:,r])
end

trial = 1:3       # single (or multiple) trial(s) for plotting (any number between 1 and R)
plot(single_indep_dt.t, single_indep_dt.y_mt[:,trial], color="black", label = "")   # raw data in black
plot!(single_indep_dt.t, y_pred[:,trial], color="red", label = "")                  # prediction in red
plot!([NaN, NaN],[NaN, NaN], color = "black", label = "raw data")
plot!([NaN, NaN],[NaN, NaN], color = "red", label = "prediction")
xlims!(0.0, 20.0)                                                           # specify x-axis limits
xlabel!(L"t")                                                               # set axis titles
ylabel!(L"Y_{ir}(t)")
title!("Single-level NLME (independent): Trial-specific predictions")       # add plot title




###########################################################################################
## 1.2 One run of single-level NLME simulation with unstructured random effects
###########################################################################################

# 1.2.1 generate a 200-trial simulated trial-level EEG data set for single-level NLME ######
## specify the true trial-level covariance matrix
V_diag = [1.5,4.0,2.0,6.0,10.0]                                     # specify the diagonl terms of the trial-level covariance matrix
V_true = zeros(Float64, 5,5)
for i = 1:5
    for j = 1:5
        if i == j
            V_true[i,j] = V_diag[i]                                 
        else
            V_true[i,j] = 0.5 * sqrt(V_diag[i] * V_diag[j])         # impose correlations between parameters
        end
    end
end
## generate a 200-trial simulated trial-level EEG data set
single_unstr_dt = nlme_unstr_generate(200,                               # R: number of trials
                                     [6.0, 5.0, 6.0, 14.0, -5.0],        # beta_true: true fixed effects
                                     V_true,   # V_true: true covariance matrix of trial-level random effects
                                     8.0,                                # sigma2_ture: true variance of measurement error
                                     collect(0:0.8:20))                  # t0: trial time grid


# 1.2.2 calculate starting values of model components ########################################
start_nlme_unstr!(singl_unstr_dt,           # m: `NlmeModel` object containing data and model components
                                            #    (list of values see in "nlme_classfiles.jl)
                    [13.0,17.0],            # peak_range: searching interval of peak-shaped component
                    [3.0,7.0])              # dip_range: searching interval of dip-shaped component


# 1.2.3 fit single-level NLME model using MM algorithm #######################################
single_unstr_mod = fit_nlme_mm_unstr!(single_unstr_dt,      # m: `NlmeModel` object containing data and model components
                                                            #    (list of values see in "nlme_classfiles.jl)
                                        20,          # gn_maxiter: maximum number of iterations of Gauss-Newton algorithm in Step 1         
                                        1e-4,        # gn_err_threshold: relative convergence threshold for Gauss-Newton algorithm
                                        20,          # halving_maxiter: maximum times of step-halving
                                        20,          # maxiter: maximum number of iterations for MM algorithm
                                        1e-4)        # logl_err_threshold: relative convergence threshold for MM algorithm

# 1.2.4 print out estimated model components #################################################
println(string("estimated fixed effects (unstructured): beta = ", single_unstr_dt.beta), "\n")
println(string("estimated trial-level variance components (unstructured): V = \n"))
println(single_unstr_dt.V)
println(string("estimated variance of measurement error (unstructured): sigma2 = ", single_unstr_dt.sigma2), "\n")

# 1.2.5 plot raw data and prediction for a single trial #######################################
y_pred = Matrix{Float64}(undef,
                size(single_unstr_dt.t,1), 
                single_unstr_dt.R)
for r in 1:single_unstr_dt.R
    y_pred[:,r] = mu(single_unstr_dt.t, single_unstr_dt.beta + single_unstr_dt.gamma[:,r])
end

trial = 1:3       # single (or multiple) trial(s) for plotting (any number between 1 and R)
plot(single_unstr_dt.t, single_unstr_dt.y_mt[:,trial], color="black", label = "")   # raw data in black
plot!(single_unstr_dt.t, y_pred[:,trial], color="red", label = "")                  # prediction in red
plot!([NaN, NaN],[NaN, NaN], color = "black", label = "raw data")
plot!([NaN, NaN],[NaN, NaN], color = "red", label = "prediction")
xlims!(0.0, 20.0)                                                           # specify x-axis limits
xlabel!(L"t")                                                               # set axis titles
ylabel!(L"Y_{ir}(t)")
title!("Single-level NLME (unstructured): Trial-specific predictions")       # add plot title







###########################################################################################
# 2. Multi-level NLME simulation described in Section 4.1
###########################################################################################
## 2.1 One run of multi-level NLME simulation with independent random effects
###########################################################################################

# 2.1.1 generate a 50-subject 50-trial-per-subject simulated data set for multi-level NLME ###
multi_indep_dt = mnlme_indep_generate(50,                           # n: number of subjects
                               50,                                  # R: number of trials per subject
                               [6.0, 5.0, 6.0, 14.0, -5.0],         # beta_true: true fixed effects
                               Diagonal([1.0,3.0,1.0,3.0,10.0]),    # U_true: true covariance matrix of subject-level random effects
                               Diagonal([1.0,5.0,1.5,6.0,15.0]),    # V_true: true covariance matrix of trial-level random effects
                               8.0,                                 # sigma2_true: true variance of measurement error
                               collect(0:0.8:20))                   # t0: trial time grid


# 2.1.2 calculate starting values of model components ########################################
start_mnlme_indep!(multi_indep_dt,      # m: 'MnlmeModel' object containing data and model components
                                        #    (list of values seee in "nlme_classfiles.jl")
                    [13.0, 17.0],       # peak_range: searching interval of peak-shpaed component
                    [3.0, 7.0],         # dip_range: seraching interval of dip-shaped component
                    10,                 # gn_maxiter: maximum number of iterations for Gauss-Newton algorithm in single-level NLME
                    1e-4,               # gn_err_threshold: relative convergence threshold for Gauss-Newton algorithm in single-level NLME
                    20,                 # halving_maxiter: maximum times of step-halving in single-level NLME
                    20,                 # maxiter: maximum number of iterations for MM algorithm for single-level NLME
                    1e-4)               # logl_err_threshold: relative convergence threshold for MM algorithm for single-level NLME


# 2.1.3 fit single-level NLME model using MM algorithm #######################################
multi_mod_indep = fit_mnlme_indep_mm!(mult_indep_dt,    # m: 'MnlmeModel' object containing data and model components
                                                        #    (list of values seee in "nlme_classfiles.jl") 
                                        10,             # gn_maxiter: maximum number of iterations for Gauss_Newton algorithm in multi-level NLME
                                        1e-4,           # gn_err_threshold: relative convergence threshold for Gauss-Newton algorithm in multi-level NLME
                                        20,             # halving_maxiter: maximum times of step-halving in multi-level NLME
                                        20,             # maxiter: maximum number of iterations for MM algorithm for multi-level NLME
                                        1e-4)           # logl_err_threshold: relative convergence threshold for MM algorithm for multi-level NLME


# 2.1.4 print out estimated model components #################################################
println(string("estimated fixed effects: beta = ", multi_indep_dt.beta), "\n")
println(string("estimated subject-level variance components: U = diag(", diag(multi_indep_dt.U) ,")\n"))
println(string("estimated trial-level variance components: V = diag(", diag(multi_indep_dt.V) ,")\n"))
println(string("estimated variance of measurement error: sigma2 = ", multi_indep_dt.sigma2), "\n")

# 2.1.5 plot raw data and prediction for a single trials #######################################
y_pred_array = Array{Matrix{Float64}}(undef, multi_indep_dt.n)    # generate array of matrices of length n

for i in 1:multi_dt.n
    # set dimension of prediction matrix (T: number of trial time point; R_i: number of trials of i-th subject)
    y_pred_array[i] = zeros(Float64, multi_indep_dt.T, multi_indep_dt.R_array[i])
    phi_i = multi_mod_indep.beta + multi_mod_indep.alpha[:,i]
    for r in 1:multi_indep_dt.R_array[i]
        y_pred_array[i][:,r] = mu(multi_indep_dt.t, phi_i + multi_indep_dt.gamma[i][:,r])
    end
end

subject = 1       # one subject for plotting (i-th subject)
trial = 1:3       # single(or multiple) trial(s) for plotting (any number between 1 and R_i)
plot(multi_indep_dt.t, multi_indep_dt.y_mt_array[subject][:,trial], color="black", label = "")      # raw data in black
plot!(multi_indep_dt.t, y_pred_array[subject][:,trial], color="red", label = "")                    # prediction in red
plot!([NaN, NaN],[NaN, NaN], color = "black", label = "raw data")                       # add legends
plot!([NaN, NaN],[NaN, NaN], color = "red", label = "prediction")
xlims!(0.0, 20.0)                                                                       # specify x-axis limits
xlabel!(L"t")                                                                           # set axis titles
ylabel!(L"Y_{ir}(t)")
title!("Multi-level NLME (independent): Trial-specific predictions")                                  # add plot title


###########################################################################################
## 2.2 One run of multi-level NLME simulation with unstructured random effects
###########################################################################################
# 2.2.1 generate a 50-subject 50-trial-per-subject simulated data set for multi-level NLME ###
## specify the true subject- and trial-level covariance matrices
U_diag = [1.0,3.0,1.0,3.0,10.0]                                 # specify the diagonal terms of the covariance matrices
V_diag = [1.0,5.0,1.5,6.0,15.0]
U_true = zeros(Float64, 5, 5)
V_true = zeros(Float64, 5, 5)
for i = 1:5
    for j = 1:5
        if i == j
            U_true[i,j] = U_diag[i]
            V_true[i,j] = V_diag[i]
        else
            U_true[i,j] = 0.5 * sqrt(U_diag[i] * U_diag[j])     # impose correlation onto the covariance matrices
            V_true[i,j] = 0.5 * sqrt(V_diag[i] * V_diag[j])
        end
    end
end
## generate the multi-level trial-level ERP dataset
multi_unstr_dt = mnlme_unstr_generate(50,                       # n: number of subjects
                               50,                              # R: number of trials per subject
                               [6.0, 5.0, 6.0, 14.0, -5.0],     # beta_true: true fixed effects
                               U_true,                          # U_true: true covariance matrix of subject-level random effects
                               V_true,                          # V_true: true covariance matrix of trial-level random effects
                               8.0,                             # sigma2_true: true variance of measurement error
                               collect(0:0.8:20))               # t0: trial time grid

# 2.2.2 calculate starting values of model components ########################################
start_mnlme_unstr!(multi_unstr_dt,      # m: 'MnlmeUnstrModel' object containing data and model components
                                        #    (list of values seee in "nlme_classfiles.jl")
                    [13.0, 17.0],       # peak_range: searching interval of peak-shpaed component
                    [3.0, 7.0],         # dip_range: seraching interval of dip-shaped component
                    10,                 # gn_maxiter: maximum number of iterations for Gauss-Newton algorithm in single-level NLME
                    1e-4,               # gn_err_threshold: relative convergence threshold for Gauss-Newton algorithm in single-level NLME
                    20,                 # halving_maxiter: maximum times of step-halving in single-level NLME
                    20,                 # maxiter: maximum number of iterations for MM algorithm for single-level NLME
                    1e-4)               # logl_err_threshold: relative convergence threshold for MM algorithm for single-level NLME


# 2.2.3 fit single-level NLME model using MM algorithm #######################################
multi_mod_unstr = fit_mnlme_unstr_mm!(mult_unstr_dt,    # m: 'MnlmeUnstrModel' object containing data and model components
                                                        #    (list of values seee in "nlme_classfiles.jl") 
                                        10,             # gn_maxiter: maximum number of iterations for Gauss_Newton algorithm in multi-level NLME
                                        1e-4,           # gn_err_threshold: relative convergence threshold for Gauss-Newton algorithm in multi-level NLME
                                        20,             # halving_maxiter: maximum times of step-halving in multi-level NLME
                                        20,             # maxiter: maximum number of iterations for MM algorithm for multi-level NLME
                                        1e-4)           # logl_err_threshold: relative convergence threshold for MM algorithm for multi-level NLME


# 2.2.4 print out estimated model components #################################################
println(string("estimated fixed effects: beta = ", multi_unstr_dt.beta), "\n")
println(string("estimated subject-level variance components: U = \n"))
println(multi_unstr_dt.U)
println(string("estimated trial-level variance components: V = \n"))
println(multi_unstr_dt.V)
println(string("estimated variance of measurement error: sigma2 = ", multi_unstr_dt.sigma2), "\n")

# 2.2.5 plot raw data and prediction for a single trials #######################################
y_pred_array = Array{Matrix{Float64}}(undef, multi_unstr_dt.n)    # generate array of matrices of length n

for i in 1:multi_dt.n
    # set dimension of prediction matrix (T: number of trial time point; R_i: number of trials of i-th subject)
    y_pred_array[i] = zeros(Float64, multi_unstr_dt.T, multi_unstr_dt.R_array[i])
    phi_i = multi_mod_unstr.beta + multi_mod_unstr.alpha[:,i]
    for r in 1:multi_unstr_dt.R_array[i]
        y_pred_array[i][:,r] = mu(multi_unstr_dt.t, phi_i + multi_unstr_dt.gamma[i][:,r])
    end
end

subject = 1       # one subject for plotting (i-th subject)
trial = 1:3       # single(or multiple) trial(s) for plotting (any number between 1 and R_i)
plot(multi_unstr_dt.t, multi_unstr_dt.y_mt_array[subject][:,trial], color="black", label = "")      # raw data in black
plot!(multi_unstr_dt.t, y_pred_array[subject][:,trial], color="red", label = "")                    # prediction in red
plot!([NaN, NaN],[NaN, NaN], color = "black", label = "raw data")                       # add legends
plot!([NaN, NaN],[NaN, NaN], color = "red", label = "prediction")
xlims!(0.0, 20.0)                                                                       # specify x-axis limits
xlabel!(L"t")                                                                           # set axis titles
ylabel!(L"Y_{ir}(t)")
title!("Multi-level NLME (unstructured): Trial-specific predictions")                   # add plot title


