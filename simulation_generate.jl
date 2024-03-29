###########################################################################################
## Description: Functions for generating simulated single- and multi-leve data described
##              in Section 4 of 'Modeling intra-individual inter-trial EEG response 
##              variability in Autism  Spectrum Disorder'.
###########################################################################################
## List of contents:
##  1. nlme_indep_generate: Function to generate a dataset with trial-level ERP responses fitted
##                          by a single-level NLME with independent random effects
##  2. nlmeunstr_generate: Function to generate a dataset with trial-level ERP responses
##                         fitted by a single-level NLME with unstructured random effects
##  3. mnlme_indep_generate: Function to generate a dataset with trial-level ERP responses fitted
##                           by a multi-level NLME with independent random effects
##  4. mnlme_unstr_generate: Function to generate a dataset with trial-level ERP responses
##                           fitted by a multi-level NLME with unstructured random effects
###########################################################################################


function nlme_indep_generate(
    R::Int64,                                       # total number of trials
    beta_true::Vector{FloatType},                   # true fixed effects (vector of length p)
    V_true::Diagonal{FloatType, Vector{FloatType}}, # true trial-level variance components (diagonal matrix of dimension p*p)
    sigma2_true::FloatType,                         # true measurement error variance
    t0::Vector{FloatType}                           # trial time grid (vector of length T)
    ) where FloatType <: AbstractFloat 

    #########################################################################################
    ## Description: Function for generating a dataset with trial-level ERP responses that fits 
    ##              single-level NLME model with independent random effects described in 
    ##              Section 4.1 of `Modeling intra-individual inter-trial EEG response variability
    ##              in Autism Spectrum Disorder`.
    ## Definition:  T: number of trial time points
    ##              R: number of trials
    ##              p: number of trial-level variance components (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Returns:     an object of `NlmeModel` type that contains generated dataset and
    ##              model components for fitting MM algorithm for single-level NLME
    ##              (list of values given in "NLME_classfiles.jl")
    #########################################################################################

    T = size(t0, 1)

    # generate trial-level random effects (gamma_true: matrix of dimension p*R)
    gamma_distribution_true = MvNormal([0,0,0,0,0], V_true) 
    gamma_true = rand(gamma_distribution_true, R)

    # generate trial-level ERP responses (y_0: matrix of dimension T*R)
    phi_true = beta_true .+ gamma_true
    y_0 = zeros(Float64, T, R)
    for i in 1:R
        y_0[:,i] = mu(t0, phi_true[:,i])
    end

    # generate mesaurment error (e_matrix: matrix of dimension T*R)
    e_distribution_true = Normal(0,sqrt(sigma2_true))
    e_matrix = rand(e_distribution_true, T, R)

    # construct `NlmeModel` type object using generated data
    y_test = y_0 + e_matrix
    test_data = NlmeModel_construct(y_test, t0, 5)
    return test_data
end





function nlme_unstr_generate(
    R::Int64,                                       # total number of trials
    beta_true::Vector{FloatType},                   # true fixed effects (vector of length p)
    V_true::Matrix{FloatType},                      # true trial-level covariance matrix (matrix of dimension p*p)
    sigma2_true::FloatType,                         # true measurement error variance
    t0::Vector{FloatType}                           # trial time grid (vector of length T)
    ) where FloatType <: AbstractFloat 

    #########################################################################################
    ## Description: Function for generating a dataset with trial-level ERP responses that fits 
    ##              single-level NLME model with unstructured random effects described in 
    ##              Section 4.1 of `Modeling intra-individual inter-trial EEG response variability
    ##              in Autism Spectrum Disorder`.
    ## Definition:  T: number of trial time points
    ##              R: number of trials
    ##              p: dimension of trial-level covariance matrix (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Returns:     an object of `NlmeModel` type that contains generated dataset and
    ##              model components for fitting MM algorithm for single-level NLME
    ##              (list of values given in "NLME_classfiles.jl")
    #########################################################################################

    T = size(t0, 1)

    # generate trial-level random effects (gamma_true: matrix of dimension p*R)
    gamma_distribution_true = MvNormal([0,0,0,0,0], V_true) 
    gamma_true = rand(gamma_distribution_true, R)

    # generate trial-level ERP responses (y_0: matrix of dimension T*R)
    phi_true = beta_true .+ gamma_true
    y_0 = zeros(Float64, T, R)
    for i in 1:R
        y_0[:,i] = mu(t0, phi_true[:,i])
    end

    # generate mesaurment error (e_matrix: matrix of dimension T*R)
    e_distribution_true = Normal(0,sqrt(sigma2_true))
    e_matrix = rand(e_distribution_true, T, R)

    # construct `NlmeUnstrModel` type object using generated data
    y_test = y_0 + e_matrix
    test_data = NlmeUnstrModel_construct(y_test, t0, 5)
    return test_data
end





function mnlme_indep_generate(
    N::Int64,                                       # total number of subjects
    R::Int64,                                       # number of trials per subject
    beta_true::Vector{FloatType},                   # true shape parameter (vector of length p)
    U_true::Diagonal{FloatType, Vector{FloatType}}, # true subject-level variance components (diagonal matrix of dimension p*p)
    V_true::Diagonal{FloatType, Vector{FloatType}}, # true trial-level variance components (diagonal matrix of dimension p*p)
    sigma2_true::FloatType,                         # true measurement error variance
    t0::Vector{FloatType}                           # trial time grid (vector of length T)
    ) where FloatType <: AbstractFloat 

    #########################################################################################
    ## Description: Function for generating a dataset with trial-level ERP responses that fits 
    ##              multi-level NLME model descripted in Section 4.2 of `Modeling intra 
    ##              -individual inter-trial EEG response variability in Autism Spectrum Disorder`.
    ## Definition:  T: number of trial time points
    ##              n: number of subjects
    ##              R: number of trials per subject
    ##              p: number of shape parameters in shape function
    ## Args:        see above
    ## Returns:     an object of `MnlmeModel` type that contains generated dataset and
    ##              model components for fitting MM algorithm for multi-level NLME
    ##              (list of values given in "NLME_classfiles.jl")
    #########################################################################################   

    T = size(t0, 1)

    # generate subject-level random effects (alpha_true: matrix of dimension p*n)
    alpha_distribution_true = MvNormal([0,0,0,0,0], U_true)
    alpha_true = rand(alpha_distribution_true, N)
    phi_true = beta_true .+ alpha_true
    
    # generate trial-level random effects (gamma_true: array of length n, each element is a matrix of dimension p*R)
    gamma_true = Array{Matrix{FloatType}}(undef, N)
    gamma_distribution_true = MvNormal([0,0,0,0,0], V_true)

    # generate trial-level ERP responses (y_0: array of length n, each element is a matrix of dimension T*R)
    y_0 = Array{Matrix{FloatType}}(undef, N)
    y_test = Array{Matrix{FloatType}}(undef, N)
    
    e_distribution_true = Normal(0,sqrt(sigma2_true))
    for i in 1:N
        y_0[i] = zeros(T, R)
        gamma_true[i] = rand(gamma_distribution_true, R)
        beta_alpha_i = phi_true[:,i]
        for j in 1:R
            # trial-spcific true shape parameters
            phi_true_j = beta_alpha_i + gamma_true[i][:,j]

            # trial-specific 'true' ERP response without noise
            y_0[i][:,j] .= mu(t0, phi_true_j)
        end
        # generate measurement error (e_matrix: matrix of dimension T*R)
        e_matrix = rand(e_distribution_true, T, R)
        y_test[i] = y_0[i] + e_matrix
    end

    # construct 'MnlmeModel' type object using generated data
    test_data = MnlmeModel_construct(y_test, t0, 5)
    return test_data
end




function mnlme_unstr_generate(
    N::Int64,                                       # total number of subjects
    R::Int64,                                       # number of trials per subject
    beta_true::Vector{FloatType},                   # true shape parameter (vector of length p)
    U_true::Matrix{FloatType},                      # true subject-level covariance matrix (matrix of dimension p*p)
    V_true::Matrix{FloatType},                      # true trial-level covariance matrix (matrix of dimension p*p)
    sigma2_true::FloatType,                         # true measurement error variance
    t0::Vector{FloatType}                           # trial time grid (vector of length T)
    ) where FloatType <: AbstractFloat 

    #########################################################################################
    ## Description: Function for generating a dataset with trial-level ERP responses that fits 
    ##              multi-level NLME model descripted in Section 4.2 of `Modeling intra 
    ##              -individual inter-trial EEG response variability in Autism Spectrum Disorder`.
    ## Definition:  T: number of trial time points
    ##              n: number of subjects
    ##              R: number of trials per subject
    ##              p: number of shape parameters in shape function
    ## Args:        see above
    ## Returns:     an object of `MnlmeModel` type that contains generated dataset and
    ##              model components for fitting MM algorithm for multi-level NLME
    ##              (list of values given in "NLME_classfiles.jl")
    #########################################################################################   

    T = size(t0, 1)

    # generate subject-level random effects (alpha_true: matrix of dimension p*n)
    alpha_distribution_true = MvNormal([0,0,0,0,0], U_true)
    alpha_true = rand(alpha_distribution_true, N)
    phi_true = beta_true .+ alpha_true
    
    # generate trial-level random effects (gamma_true: array of length n, each element is a matrix of dimension p*R)
    gamma_true = Array{Matrix{FloatType}}(undef, N)
    gamma_distribution_true = MvNormal([0,0,0,0,0], V_true)

    # generate trial-level ERP responses (y_0: array of length n, each element is a matrix of dimension T*R)
    y_0 = Array{Matrix{FloatType}}(undef, N)
    y_test = Array{Matrix{FloatType}}(undef, N)
    
    e_distribution_true = Normal(0,sqrt(sigma2_true))
    for i in 1:N
        y_0[i] = zeros(T, R)
        gamma_true[i] = rand(gamma_distribution_true, R)
        beta_alpha_i = phi_true[:,i]
        for j in 1:R
            # trial-spcific true shape parameters
            phi_true_j = beta_alpha_i + gamma_true[i][:,j]

            # trial-specific 'true' ERP response without noise
            y_0[i][:,j] .= mu(t0, phi_true_j)
        end
        # generate measurement error (e_matrix: matrix of dimension T*R)
        e_matrix = rand(e_distribution_true, T, R)
        y_test[i] = y_0[i] + e_matrix
    end

    # construct 'MnlmeModel' type object using generated data
    test_data = MnlmeUnstrModel_construct(y_test, t0, 5)
    return test_data
end


