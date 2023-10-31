function NLME_single_generate(
    R::Int64,                                       # total number of trials
    beta_true::Vector{FloatType},                   # the true fixed effects (vector of length p)
    V_true::Diagonal{FloatType, Vector{FloatType}}, # the true trial-level variance components (diagonal matrix of dimension p*p)
    sigma2_true::FloatType,                         # the true measurement error variance
    t0::Vector{FloatType}                           # the trial time grid (vector of length T)
    ) where FloatType <: AbstractFloat 

    #########################################################################################
    ## Description: Function for generating a dataset with trial-level ERP responses that fits 
    ##              the single-level NLME model descripted in Section 4.1 of `Modeling intra 
    ##              -individual inter-trial EEG response variability in Autism Spectrum Disorder`.
    ## Definition:  T: number of trial time points
    ##              R: number of trials
    ##              p: number of trial-level variance components (equals to the number of shape 
    ##                 parameters in the shape function)
    ## Args:        see above
    ## Returns:     an object of `NlmeModel` type that contains the generated dataset and
    ##              and suitable for the main function of 'fit_nlme_mm!'
    #########################################################################################

    T = size(t0, 1)

    # generate the trial-level random effects (alpha_true: matrix of dimension p*R)
    alpha_distribution_true = MvNormal([0,0,0,0,0], V_true) 
    alpha_true = rand(alpha_distribution_true, R)

    # generate the trial-level ERP responses (y_0: matrix of dimension T*R)
    phi_true = beta_true .+ alpha_true
    y_0 = zeros(Float64, T, R)
    for i in 1:R
        y_0[:,i] = mu(t0, phi_true[:,i])
    end

    # generate the mesaurment error (e_matrix: matrix of dimension T*R)
    e_distribution_true = Normal(0,sqrt(sigma2_true))
    e_matrix = rand(e_distribution_true, T, R)

    # construct the `NlmeModel` type object using the generated data
    y_test = y_0 + e_matrix
    test_data = NlmeModel_construct(y_test, t0, 5)
    return test_data
end

