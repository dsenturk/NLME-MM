###########################################################################################
## Description: Type definition and construction functions for `NlmeModel` used in fitting
##              single- and multi-level NLME models with independent and unstructured
##              random effects via the proposed algorithm.
###########################################################################################
## List of contents:
##    1. NlmeModel: A type that contains model components and placeholders for variables
##                  used in MM algorithm for single-level NLME model with independent
##                  random effects
##    2. NlmeModel_construct: Function that constructs an object of 'NlmeModel' type
##    3. MnlmeModel: A type that contains model components and placeholders for variables
##                   used in MM algorithm for multi-level NLME model with independent
##                   random effects
##    4. NlmeModel_construct: Function that constructs an object of 'MnlmeModel' type
##    5. NlmeUnstrModel: A type that contains model components and placeholders for 
##                       variables used in MM algorithm for single-level NLME model with
##                       unstructured random effects
##    6. NlmeUnstrModel_construct: Function that constructs an object of 'NlmeModel' type
##    7. MnlmUnstreModel: A type that contains model components and placeholders for 
##                        variables used in MM algorithm for multi-level NLME model with
##                        unstructured random effects
##    8. NlmeUnstrModel_construct: Function that constructs an object of 'MnlmeModel' type
###########################################################################################


mutable struct NlmeModel{FloatType <: AbstractFloat}
    #########################################################################################
    ## Description: Type that contains model components and placeholders for variables used
    ##              in MM algorithm for single-level NLME model
    ## Definition:  T: number of trial time points
    ##              R: number of trials
    ##              p: number of trial-level variance components (equals to number of shape 
    ##                 parameters in shape function) 
    #########################################################################################
    # data
    y_mt::Matrix{FloatType}                     # trial-level ERP responses (matrix of dimension T*R)
    t::Vector{FloatType}                        # trial time grid (vector of length T)

    # model components
    beta::Vector{FloatType}                     # current estimation of fixed effects (vector of length p)
    gamma::Matrix{FloatType}                    # current estimation of random effects (matrix of dimension p*R)
    V::Diagonal{FloatType, Vector{FloatType}}   # current estimation of covariance matrix of trial-level random effects
                                                #   (diagonal matrix of dimension p*p)
    sigma2::FloatType                           # current estimation of variance of measurement error
    logl::FloatType                             # approximated marginal log-likelihood evaluated at current estimation
                                                #   of model components
    
    # placeholders for better memory allocation
    R::Int64                                    # total number of trials
    V_half::Diagonal{FloatType, Vector{FloatType}}
    V_inv::Diagonal{FloatType, Vector{FloatType}}
    Delta::Diagonal{FloatType, Vector{FloatType}}
    ## variables in Step 1
    gn_iter::Int
    gn_err::FloatType
    pnls_w::FloatType
    pnls_w1::FloatType
    halv_iter::Int
    halv_err::FloatType
    beta_w::Vector{FloatType}
    beta_w1::Vector{FloatType}
    delta_beta::Vector{FloatType}
    gamma_w::Matrix{FloatType}
    gamma_w1::Matrix{FloatType}
    delta_gamma::Matrix{FloatType}
    R_1::SharedMatrix{FloatType}
    S_1::SharedMatrix{FloatType}
    s_1::SharedArray{FloatType}
    S_0::SharedMatrix{FloatType}
    s_0::SharedArray{FloatType}
    phi_r::Vector{FloatType}
    M_r::Matrix{FloatType}
    w_r::Vector{FloatType}
    K_r::Matrix{FloatType}
    R_11_r::Matrix{FloatType}
    Q_r:: LinearAlgebra.QRCompactWYQ{FloatType, Matrix{FloatType}, Matrix{FloatType}}
    R_r_right::Matrix{FloatType}
    ## variables in Step 2
    iter::Int
    sigma_num::FloatType
    sigma_den::FloatType
    V_num::Vector{FloatType}
    V_den::Vector{FloatType}
    logl_pre::FloatType
    logl_err::FloatType
    omega_r_inv::Matrix{FloatType}
    mm_update_value::SharedMatrix{FloatType}
    mm_update_sum::Matrix{FloatType}
end



function NlmeModel_construct(
    y_mt::Matrix{FloatType},            # trial-level ERP responses (matrix of dimension T*R)
    t::Vector{FloatType},               # trial time grid (vector of length T)
    p::Int                              # number of shape parameters
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that contructs object of type `NlmeModel` that contains trial-level
    ##              ERP responses, model components, and placeholders for variables used in the
    ##              MM algorithm for single-level NLME
    ## Definition:  T: number of trial time points
    ##              R: number of trials
    ##              p: number of trial-level variance components (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Returns:     `NlmeModel` type object, input to main function of 'fit_nlme_mm!'
    #########################################################################################

    # set type and dimension of all components in `NlmeModel` object
    T = size(y_mt, 1)
    R = size(y_mt, 2)
    beta = Vector{FloatType}(undef, p)
    gamma = Matrix{FloatType}(undef, p, R)
    V = Diagonal(Vector{FloatType}(undef, p))
    sigma2 = one(FloatType)
    logl = -1e8
    V_half = Diagonal(Vector{FloatType}(undef, p))
    V_inv = Diagonal(Vector{FloatType}(undef, p))
    Delta = Diagonal(Vector{FloatType}(undef, p))
    gn_iter = 1
    gn_err = convert(FloatType, 10.0)
    pnls_w = zero(FloatType)
    pnls_w1 = zero(FloatType)
    halv_iter = 1
    halv_err = convert(FloatType, 10.0)
    beta_w = copy(beta)
    beta_w1 = copy(beta)
    delta_beta = copy(beta)
    gamma_w = copy(gamma)
    gamma_w1 = copy(gamma)
    delta_gamma = copy(gamma)
    R_1 = SharedMatrix{FloatType}(R*p, p)
    S_1 = SharedMatrix{FloatType}(R*p, p)
    s_1 = SharedArray{FloatType}(R*p)
    S_0 = SharedMatrix{FloatType}(R*T, p)
    s_0 = SharedArray{FloatType}(R*T)
    phi_r = Vector{FloatType}(undef, p)
    M_r = Matrix{FloatType}(undef, T, p)
    w_r = Vector{FloatType}(undef, T)
    K_r = Matrix{FloatType}(undef, T+p, p)
    R_11_r = Matrix{FloatType}(undef, p, p)
    Q_r = LinearAlgebra.QRCompactWYQ{FloatType, Matrix{FloatType}, Matrix{FloatType}}(K_r, R_11_r)
    R_r_right = Matrix{FloatType}(undef, T+p, p+1)
    iter = 1
    sigma_num = one(FloatType)
    sigma_den = one(FloatType)
    V_num = Vector{FloatType}(undef, p)
    V_den = Vector{FloatType}(undef, p)
    logl_pre = -1e8
    logl_err = convert(FloatType, 10.0)
    omega_r_inv = Matrix{FloatType}(undef, T, T)
    mm_update_value = SharedMatrix{FloatType}(2*p+3, R)
    mm_update_sum = Matrix{FloatType}(undef, 2*p+3, 1)

    return NlmeModel{FloatType}(y_mt, t, beta, gamma, V, sigma2, logl, R, V_half, V_inv, Delta,
                       gn_iter, gn_err, pnls_w, pnls_w1, halv_iter, halv_err,
                       beta_w, beta_w1, delta_beta, gamma_w, gamma_w1, delta_gamma,
                       R_1, S_1, s_1, S_0, s_0,
                       phi_r, M_r, w_r, K_r, R_11_r, Q_r,
                       R_r_right,
                       iter, sigma_num, sigma_den, V_num, V_den, logl_pre, logl_err,
                       omega_r_inv, mm_update_value, mm_update_sum)
end



mutable struct MnlmeModel{FloatType <: AbstractFloat}
    #########################################################################################
    ## Description: Type that contains model components and placeholders for variables used
    ##              in MM algorithm for multi-level NLME model with independent random effects
    ## Definition:  T: number of trial time points
    ##              n: number of subjects
    ##              R_i: number of trials, i = 1,..., n
    ##              p: number of variance components on each level (equals to number of shape 
    ##                 parameters in shape function) 
    #########################################################################################
    # data
    y_mt_array::Array{Matrix{FloatType}}        # trial-level ERP responses for each subject 
                                                # (array of matrices of dimension T*R_i, i = 1,...,n)
    t::Vector{FloatType}                        # trial time grid (vector of length T)
    T::Int64                                    # number of trial time points
    p::Int64                                    # number of shape parameters
    n::Int64                                    # number of subjects
    R_array::Array{Int64}                       # number of trials of each subject (vector of length n: {R_1,...,R_n})
    R::Int64                                    # total number of trials (R = sum_{i=1}^{n}R_i)
    # model components
    beta::Vector{FloatType}                     # current estimation of fixed effects (vector of length p)
    alpha::Matrix{FloatType}                    # current estimation of subject-level random effects 
                                                # (matrix of dimension p*n)
    gamma::Array{Matrix{FloatType}}             # current estimation of trial-level random effects for each subject
                                                # (array of matrices of dimension p*R_i, i = 1,...,n)
    U::Diagonal{FloatType,Vector{FloatType}}    # current estimation of  covariance matrix of subject-level random effects 
                                                # (diagonal matrix of dimension p*p)
    V::Diagonal{FloatType,Vector{FloatType}}    # current estimation of covariance matrix of trial-level random effects 
                                                # (diagonal matrix of dimension p*p)
    sigma2::FloatType                           # variance of measurement errors
    logl::FloatType                             # approximated marginal log-liklihood eavluated at current estimation 
                                                # of model components
    # place holders for better memory allocation
    U_half::Diagonal{FloatType,Vector{FloatType}}
    U_inv::Diagonal{FloatType,Vector{FloatType}}
    Delta_U::Diagonal{FloatType,Vector{FloatType}}
    V_half::Diagonal{FloatType,Vector{FloatType}}
    V_inv::Diagonal{FloatType,Vector{FloatType}}
    Delta_V::Diagonal{FloatType,Vector{FloatType}}
    ## variables in Step 1
    gn_iter::Int64
    gn_err::FloatType
    pnls_w::FloatType
    pnls_w1::FloatType
    beta_w::Vector{FloatType}
    beta_w1::Vector{FloatType}
    delta_beta::Vector{FloatType}
    alpha_w::Matrix{FloatType}
    alpha_w1::Matrix{FloatType}
    delta_alpha::Matrix{FloatType}
    gamma_w::Array{Matrix{FloatType}}
    gamma_w1::Array{Matrix{FloatType}}
    delta_gamma::Array{Matrix{FloatType}}
    s_1_ir::Matrix{FloatType}
    S_beta_1_ir::Matrix{FloatType}
    S_alpha_1_ir::Matrix{FloatType}
    R_1_ir::Matrix{FloatType}
    s_01_i::Matrix{FloatType}
    S_beta_01_i::Matrix{FloatType}
    R_2_i::Matrix{FloatType}
    s_0::Matrix{FloatType}
    S_beta_00::Matrix{FloatType}
    iter::Int64
    ## variables in Step 2
    sigma_num::FloatType
    sigma_den::FloatType
    U_num::Vector{FloatType}
    U_den::Vector{FloatType}
    V_num::Vector{FloatType}
    V_den::Vector{FloatType}
    logl_pre::FloatType
    logl_err::FloatType
    mm_update_value::Matrix{FloatType}
    mm_update_sum::Matrix{FloatType}
end


function MnlmeModel_construct(
    y_mt_array::Array{Matrix{FloatType}},       # trial-level ERP responses for each subject (array of matrices of dimension T*R_i, i = 1,...,N)
    t::Vector{FloatType},                       # trial time grid (vector of length T)
    p::Int64                                    # number of shape parameters
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that contructs object of type `MnlmeModel` that contains trial-
    ##              level ERP responses, model components, and placeholders for variables used
    ##              in MM algorithm for multi-level NLME with independent random effects
    ## Definition:  T: number of trial time points
    ##              n: number of subjects
    ##              R_i: number of trials for subject i, i = 1,...,n
    ##              p: number of variance components on each level (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Returns:     `MnlmeModel` type object, input to main function of 'fit_mnlme_mm!'
    #########################################################################################

    # set up types and dimensions of all components in "MnlmeModel"
    T = size(t, 1)
    n = size(y_mt_array, 1)
    R_array = Vector{Int64}(undef, n)
    @inbounds for i=1:n
        R_array[i] = size(y_mt_array[i], 2)
    end
    R = sum(R_array)
    # model components
    beta = Vector{FloatType}(undef, p)
    alpha = Matrix{FloatType}(undef, p, n)
    gamma = Array{Matrix{FloatType}}(undef, n)
    gamma_w = Array{Matrix{FloatType}}(undef, n)
    gamma_w1 = Array{Matrix{FloatType}}(undef, n)
    delta_gamma = Array{Matrix{FloatType}}(undef, n)
    @inbounds for i=1:n
        gamma[i] = Matrix(undef, p, R_array[i])
        gamma_w[i] = Matrix(undef, p, R_array[i])
        gamma_w1[i] = Matrix(undef, p, R_array[i])
        delta_gamma[i] = Matrix(undef, p, R_array[i])
    end
    U = Diagonal(Vector{FloatType}(undef, p))
    V = Diagonal(Vector{FloatType}(undef, p))
    sigma2 = one(FloatType)
    logl = -1e8
    # placeholders for better memory allocation
    U_half = Diagonal(Vector{FloatType}(undef, p))
    U_inv = Diagonal(Vector{FloatType}(undef, p))
    Delta_U = Diagonal(Vector{FloatType}(undef, p))
    V_half = Diagonal(Vector{FloatType}(undef, p))
    V_inv = Diagonal(Vector{FloatType}(undef, p))
    Delta_V = Diagonal(Vector{FloatType}(undef, p))
    ## variables for Step 1
    gn_iter = 1
    gn_err = convert(FloatType, 10.0)
    pnls_w = zero(FloatType)
    pnls_w1 = zero(FloatType)
    beta_w = copy(beta)
    beta_w1 = copy(beta)
    delta_beta = copy(beta)
    alpha_w = copy(alpha)
    alpha_w1 = copy(alpha)
    delta_alpha = copy(alpha)
    s_1_ir = Matrix{FloatType}(undef, R*p, 1)
    S_beta_1_ir = Matrix{FloatType}(undef, R*p, p)
    S_alpha_1_ir = Matrix{FloatType}(undef, R*p, p)
    R_1_ir = Matrix{FloatType}(undef, R*p, p)
    s_01_i = Matrix{FloatType}(undef, n*p, 1)
    S_beta_01_i = Matrix{FloatType}(undef, n*p, p)
    R_2_i = Matrix{FloatType}(undef, n*p, p)
    s_0 = Matrix{FloatType}(undef, T*n, 1)
    S_beta_00 = Matrix{FloatType}(undef, T*n, p)
    ## Variables for Step 2
    iter = 1
    sigma_num = one(FloatType)
    sigma_den = one(FloatType)
    U_num = Vector{FloatType}(undef, p)
    U_den = Vector{FloatType}(undef, p)
    V_num = Vector{FloatType}(undef, p)
    V_den = Vector{FloatType}(undef, p)
    logl_pre = -1e8
    logl_err = convert(FloatType, 10.0)
    mm_update_value = Matrix{FloatType}(undef, 4*p+3, n)
    mm_update_sum = Matrix{FloatType}(undef, 4*p+3, 1)
return MnlmeModel{FloatType}(y_mt_array, t, T, p, n, R_array, R,
                    beta, alpha, gamma, U, V, sigma2, logl,
                    U_half, U_inv, Delta_U, V_half, V_inv, Delta_V,
                    gn_iter, gn_err, pnls_w, pnls_w1,
                    beta_w, beta_w1, delta_beta,
                    alpha_w, alpha_w1, delta_alpha,
                    gamma_w, gamma_w1, delta_gamma,
                    s_1_ir, S_beta_1_ir, S_alpha_1_ir, R_1_ir,
                    s_01_i, S_beta_01_i, R_2_i,
                    s_0, S_beta_00,
                    iter, sigma_num, sigma_den, U_num, U_den, V_num, V_den,
                    logl_pre, logl_err, mm_update_value, mm_update_sum)
end






mutable struct NlmeUnstrModel{FloatType <: AbstractFloat}
    #########################################################################################
    ## Description: Type that contains model components and placeholders for variables in
    ##              MM algorithm for single-level NLME model with unstructured random effects
    ## Definition:  T: number of trial time points
    ##              R: number of trials
    ##              p: dimesnion of trial-level covariance matrix (equals to number of shape 
    ##                 parameters in shape function) 
    #########################################################################################
    # data
    y_mt::Matrix{FloatType}                     # trial-level ERP responses (matrix of dimension T*R)
    t::Vector{FloatType}                        # trial time grid (vector of length T)

    # model components
    beta::Vector{FloatType}                     # current estimation of fixed effects (vector of length p)
    gamma::Matrix{FloatType}                    # current estimation of random effects (matrix of dimension p*R)
    V::Matrix{FloatType}                        # current estimation of covariance matrix of trial-level random effects
                                                # (matrix of dimension p*p)
    sigma2::FloatType                           # current estimation of variance of measurement error
    logl::FloatType                             # approximated marginal log-likelihood evaluated at current estimation
                                                #   of model components
    R::Int64                                    # total number of trials

    # placeholder for better memory allocation
    V_half::Matrix{FloatType}
    V_inv::Matrix{FloatType}
    Delta::Matrix{FloatType}
    ## variables in Step 1
    gn_iter::Int
    gn_err::FloatType
    pnls_w::FloatType
    pnls_w1::FloatType
    halv_iter::Int
    halv_err::FloatType
    beta_w::Vector{FloatType}
    beta_w1::Vector{FloatType}
    delta_beta::Vector{FloatType}
    gamma_w::Matrix{FloatType}
    gamma_w1::Matrix{FloatType}
    delta_gamma::Matrix{FloatType}
    R_1::SharedMatrix{FloatType}
    S_1::SharedMatrix{FloatType}
    s_1::SharedArray{FloatType}
    S_0::SharedMatrix{FloatType}
    s_0::SharedArray{FloatType}
    phi_r::Vector{FloatType}
    M_r::Matrix{FloatType}
    w_r::Vector{FloatType}
    K_r::Matrix{FloatType}
    R_11_r::Matrix{FloatType}
    Q_r:: LinearAlgebra.QRCompactWYQ{FloatType, Matrix{FloatType}, Matrix{FloatType}}
    R_r_right::Matrix{FloatType}
    ## variables in Step 2
    iter::Int
    sigma_num::FloatType
    sigma_den::FloatType
    logl_pre::FloatType
    logl_err::FloatType
    omega_r_inv::Matrix{FloatType}
    mm_update_value::SharedMatrix{FloatType}
    m_omega_m_series::SharedMatrix{FloatType}
    ss_series::SharedMatrix{FloatType}
    mm_update_sum::Matrix{FloatType}
    m_omega_m_sum::Matrix{FloatType}
    ss_sum::Matrix{FloatType}
end




function NlmeUnstrModel_construct(
    y_mt::Matrix{FloatType},            # trial-level ERP responses (matrix of dimension T*R)
    t::Vector{FloatType},               # trial time grid (vector of length T)
    p::Int                              # number of shape parameters
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that contructs object of type `NlmeUnstrModel` that contains 
    ##              trial-level ERP responses, model components, and placeholders for variables
    ##              used in the MM algorithm for single-level NLME with unstructured random
    ##              effects
    ## Definition:  T: number of trial time points
    ##              R: number of trials
    ##              p: dimension of trial-level covariance matrix (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Returns:     `NlmeUnstrModel` type object, input to main function of 'fit_nlme_unstr_mm!'
    #########################################################################################

    # set type and dimension of all components in `NlmeUnstrModel` object
    T = size(y_mt, 1)
    R = size(y_mt, 2)
    beta = Vector{FloatType}(undef, p)
    gamma = Matrix{FloatType}(undef, p, R)
    V = Matrix{TFloatType}(undef, p, p)
    sigma2 = one(FloatType)
    logl = -1e8
    V_half = Matrix{FloatType}(undef, p, p)
    V_inv = Matrix{FloatType}(undef, p, p)
    Delta = Matrix{FloatType}(undef, p, p)
    gn_iter = 1
    gn_err = convert(FloatType, 10.0)
    pnls_w = zero(FloatType)
    pnls_w1 = zero(FloatType)
    halv_iter = 1
    halv_err = convert(T, 10.0)
    beta_w = copy(beta)
    beta_w1 = copy(beta)
    delta_beta = copy(beta)
    gamma_w = copy(gamma)
    gamma_w1 = copy(gamma)
    delta_gamma = copy(gamma)
    R_1 = SharedMatrix{FloatType}(R*p, p)
    S_1 = SharedMatrix{FloatType}(R*p, p)
    s_1 = SharedArray{FloatType}(R*p)
    S_0 = SharedMatrix{FloatType}(R*T, p)
    s_0 = SharedArray{FloatType}(R*T)
    phi_r = Vector{FloatType}(undef, p)
    M_r = Matrix{FloatType}(undef, T, p)
    w_r = Vector{FloatType}(undef, T)
    K_r = Matrix{FloatType}(undef, T+p, p)
    R_11_r = Matrix{FloatType}(undef, p, p)
    Q_r = LinearAlgebra.QRCompactWYQ{FloatType, Matrix{FloatType}, Matrix{FloatType}}(K_r, R_11_r)
    R_r_right = Matrix{FloatType}(undef, T+p, p+1)
    iter = 1
    sigma_num = one(T)
    sigma_den = one(T)
    logl_pre = -1e8
    logl_err = convert(T, 10.0)
    omega_r_inv = Matrix{FloatType}(undef, T, T)
    mm_update_value = SharedMatrix{FloatType}(3, R)
    m_omega_m_series = SharedMatrix{FloatType}(R*p, p)
    ss_series = SharedMatrix{FloatType}(R*p, p)
    mm_update_sum = Matrix{FloatType}(undef, 3, 1)
    m_omega_m_sum = Matrix{FloatType}(undef, p, p)
    ss_sum = Matrix{FloatType}(undef, p, p)

    return NlmeUnstrModel{FloatType}(y_mt, t, beta, gamma, V, sigma2, logl, R, V_half, V_inv, Delta,
                        gn_iter, gn_err, pnls_w, pnls_w1, halv_iter, halv_err,
                        beta_w, beta_w1, delta_beta, gamma_w, gamma_w1, delta_gamma,
                        R_1, S_1, s_1, S_0, s_0,
                        phi_r, M_r, w_r, K_r, R_11_r, Q_r,
                        R_r_right,
                        iter, sigma_num, sigma_den, V_num, V_den, logl_pre, logl_err,
                        omega_r_inv, mm_update_value, m_omega_m_series, ss_series,
                        mm_update_sum, m_omega_m_sum, ss_sum)
end







mutable struct MnlmeUnstrModel{FloatType <: AbstractFloat}
    #########################################################################################
    ## Description: Type that contains model components and placeholders for variables used
    ##              in MM algorithm for multi-level NLME model with unstructured random effects
    ## Definition:  T: number of trial time points
    ##              n: number of subjects
    ##              R_i: number of trials, i = 1,..., n
    ##              p: dimension of random effects on each level (equals to number of shape 
    ##                 parameters in shape function) 
    #########################################################################################
    # data
    y_mt_array::Array{Matrix{FloatType}}        # trial-level ERP responses for each subject 
                                                # (array of matrices of dimension T*R_i, i = 1,...,n)
    t::Vector{FloatType}                        # trial time grid (vector of length T)
    T::Int64                                    # number of trial time points
    p::Int64                                    # number of shape parameters
    n::Int64                                    # number of subjects
    R_array::Array{Int64}                       # number of trials of each subject (vector of length n: {R_1,...,R_n})
    R::Int64                                    # total number of trials (R = sum_{i=1}^{n}R_i)
    # model components
    beta::Vector{FloatType}                     # current estimation of fixed effects (vector of length p)
    alpha::Matrix{FloatType}                    # current estimation of subject-level random effects 
                                                # (matrix of dimension p*n)
    gamma::Array{Matrix{FloatType}}             # current estimation of trial-level random effects for each subject
                                                # (array of matrices of dimension p*R_i, i = 1,...,n)
    U::Matrix{FloatType}                        # current estimation of  covariance matrix of subject-level random effects 
                                                # (matrix of dimension p*p)
    V::Matrix{FloatType}                        # current estimation of covariance matrix of trial-level random effects 
                                                # (matrix of dimension p*p)
    sigma2::FloatType                           # variance of measurement errors
    logl::FloatType                             # approximated marginal log-liklihood eavluated at current estimation 
                                                # of model components

    # placeholders for better memory allocation
    U_half::Matrix{FloatType}
    U_inv::Matrix{FloatType}
    Delta_U::Matrix{FloatType}
    V_half::Matrix{FloatType}
    V_inv::Matrix{FloatType}
    Delta_V::Matrix{FloatType}
    ## variables for Step 1
    gn_iter::Int64
    gn_err::FloatType
    pnls_w::FloatType
    pnls_w1::FloatType
    beta_w::Vector{FloatType}
    beta_w1::Vector{FloatType}
    delta_beta::Vector{FloatType}
    alpha_w::Matrix{FloatType}
    alpha_w1::Matrix{FloatType}
    delta_alpha::Matrix{FloatType}
    gamma_w::Array{Matrix{FloatType}}
    gamma_w1::Array{Matrix{FloatType}}
    delta_gamma::Array{Matrix{FloatType}}
    s_1_ir::Matrix{FloatType}
    S_beta_1_ir::Matrix{FloatType}
    S_alpha_1_ir::Matrix{FloatType}
    R_1_ir::Matrix{FloatType}
    s_01_i::Matrix{FloatType}
    S_beta_01_i::Matrix{FloatType}
    R_2_i::Matrix{FloatType}
    s_0::Matrix{FloatType}
    S_beta_00::Matrix{FloatType}
    iter::Int
    ## variables for Steps 2
    n_omega_n_series::Matrix{FloatType}
    rr_seris::Matrix{FloatType}
    m_omega_m_series::Matrix{FloatType}
    ss_series::Matrix{FloatType}
    mm_update_value::Matrix{TFloatType}
    n_omega_n_sum::Matrix{FloatType}
    rr_sum::Matrix{FloatType}
    m_omega_m_sum::Matrix{FloatType}
    ss_sum::Matrix{FloatType}
    mm_update_sum::Matrix{FloatType}
    logl_pre::FloatType
    logl_err::FloatType
end



function MnlmeUnstrModel_construct(
    y_mt_array::Array{Matrix{FloatType}},       # trial-level ERP responses for each subject (array of matrices of dimension T*R_i, i = 1,...,N)
    t::Vector{FloatType},                       # trial time grid (vector of length T)
    p::Int64                                    # number of shape parameters
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that contructs object of type `MnlmeUnstrModel` that contains 
    ##              trial-level ERP responses, model components, and placeholders for variables
    ##              in MM algorithm for multi-level NLME with unstructured random effects
    ## Definition:  T: number of trial time points
    ##              n: number of subjects
    ##              R_i: number of trials for subject i, i = 1,...,n
    ##              p: dimension of covariance matrix on each level (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Returns:     `MnlmeUnstrModel` type object, input to main function of 'fit_mnlmeunstr_mm!'
    #########################################################################################

    # set up types and dimensions of all components in "MnlmeUnstrModel"
    T = size(t, 1)
    n = size(y_mt_array, 1)
    R_array = Vector{Int64}(undef, n)
    @inbounds for i=1:n
        R_array[i] = size(y_mt_array[i], 2)
    end
    R = sum(R_array)
    # model components
    beta = Vector{FloatType}(undef, p)
    alpha = Matrix{FloatType}(undef, p, n)
    gamma = Array{Matrix{FloatType}}(undef, n)
    gamma_w = Array{Matrix{FloatType}}(undef, n)
    gamma_w1 = Array{Matrix{FloatType}}(undef, n)
    delta_gamma = Array{Matrix{FloatType}}(undef, n)
    @inbounds for i=1:n
        gamma[i] = Matrix(undef, p, R_array[i])
        gamma_w[i] = Matrix(undef, p, R_array[i])
        gamma_w1[i] = Matrix(undef, p, R_array[i])
        delta_gamma[i] = Matrix(undef, p, R_array[i])
    end
    U = Matrix{FloatType}(undef, p, p)
    V = Matrix{FloatType}(undef, p, p)
    sigma2 = one(FloatType)
    logl = -1e8
    # placeholders for better memory allocation
    U_half = Matrix{FloatType}(undef, p, p)
    U_inv = Matrix{FloatType}(undef, p, p)
    Delta_U = Matrix{FloatType}(undef, p, p)
    V_half = Matrix{FloatType}(undef, p, p)
    V_inv = Matrix{FloatType}(undef, p, p)
    Delta_V = Matrix{FloatType}(undef, p, p)
    ## variables for Step 1
    gn_iter = 1
    gn_err = convert(FloatType, 10.0)
    pnls_w = zero(FloatType)
    pnls_w1 = zero(FloatType)
    beta_w = copy(beta)
    beta_w1 = copy(beta)
    delta_beta = copy(beta)
    alpha_w = copy(alpha)
    alpha_w1 = copy(alpha)
    delta_alpha = copy(alpha)
    s_1_ir = Matrix{FloatType}(undef, R*p, 1)
    S_beta_1_ir = Matrix{FloatType}(undef, R*p, p)
    S_alpha_1_ir = Matrix{FloatType}(undef, R*p, p)
    R_1_ir = Matrix{FloatType}(undef, R*p, p)
    s_01_i = Matrix{FloatType}(undef, n*p, 1)
    S_beta_01_i = Matrix{FloatType}(undef, n*p, p)
    R_2_i = Matrix{FloatType}(undef, n*p, p)
    s_0 = Matrix{FloatType}(undef, T*n, 1)
    S_beta_00 = Matrix{FloatType}(undef, T*n, p)
    ## variables for Step 2
    iter = 1
    logl_pre = -1e8
    logl_err = convert(FloatType, 10.0)
    n_omega_n_series = Matrix{FloatType}(undef, n*p, p)
    rr_series = Matrix{FloatType}(undef, n*p, p)
    m_omega_m_series = Matrix{FloatType}(undef, n*p, p)
    ss_series = Matrix{FloatType}(undef, n*p, p)
    mm_update_value = Matrix{FloatType}(undef, 3, n)
    n_omega_n_sum = Matrix{FloatType}(undef, p, p)
    rr_sum = Matrix{FloatType}(undef, p, p)
    m_omega_m_sum = Matrix{FloatType}(undef, p, p)
    ss_sum = Matrix{FloatType}(undef, p, p)
    mm_update_sum = Matrix{FloatType}(undef, 3, 1)

    return MnlmeUnstrModel{FloatType}(y_mt_array, t, T, p, n, R_array, R,
                                      beta, alpha, gamma, U, V, sigma2, logl,
                                      U_half, U_inv, Delta_U, V_half, V_inv, Delta_V,
                                      gn_iter, gn_err, pnls_w, pnls_w1,
                                      beta_w, beta_w1, delta_beta,
                                      alpha_w, alpha_w1, delta_alpha,
                                      gamma_w, gamma_w1, delta_gamma,
                                      s_1_ir, S_beta_1_ir, S_alpha_1_ir, R_1_ir,
                                      s_01_i, S_beta_01_i, R_2_i,
                                      s_0, S_beta_00,
                                      iter, n_omega_n_series, rr_series, m_omega_m_series, ss_series,
                                      mm_update_value, n_omega_n_sum, rr_sum, m_omega_m_sum, ss_sum,
                                      mm_update_sum, logl_pre, logl_err)
end
