function NLME_multi_generate(
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
    ##              multi-level NLME model described in Section 4.2 of `Modeling intra 
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
            # trial-specific true shape parameters
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


