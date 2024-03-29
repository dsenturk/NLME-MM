###########################################################################################
## Description: Functions for fitting single-level NLME model with independent and 
##              unstructured random effects via MM algorithm described in 'Modeling intra-
##              individual inter-trial EEG response variability in Autism Spectrum Disorder'.
###########################################################################################
## Functions included:
## Main function:
##    1. fit_nlme_indep_mm!: Function that fits single-level NLME model with independent random effects 
##                     via MM algorithm
##    2. fit_nlme_unstr_mm!: Function that fits single-level NLME model with unstructured
##                           random effects via MM algorithm
## Supporting functions used by main function:
##    (For independent models)
##    1.1 update_mm_indep!: Function that performs one iteration of MM algorithm
##    1.2 trial_qr_parallel_indep: Function that performs QR decomposition in Step1 of each iteration
##    1.3 mm_value_update_parallel_indep: Function that calculates components for updating variance 
##                                  components in Step 2 of each iteration
##    1.4 start_nlme_indep!: Function that calculates starting values of model components
##    (For unstructured models)
##    2.1 update_mm_unstr!: Function that performs one iteration of MM algorithm
##    2.2 trial_qr_parallel_unstr: Function that performs QR decomposition in Step1 of each 
##                                 iteration
##    2.3 mm_value_update_parallel_unstr: Function that calculates components for updating variance 
##                                  components in Step 2 of each iteration
##    2.4 start_nlme_unstr!: Function that calculates starting values of model components
###########################################################################################




function fit_nlme_indep_mm!(
    m::NlmeModel{FloatType},        # NlmeModel type object constructed by NlmeModel_construct
                                    # list of values, including:
                                    #       y_mt: trial-level ERP responses (matrix of dimension T*R)
                                    #       t: trial time grid (vector of length T)
                                    #       beta: current estimates of fixed effects (vector of length p)
                                    #       gamma: current estiamtes of trial-level random effects (matrix of p*R)
                                    #       V: current estimates of trial-level variance components (diagonal matrix of dimension p*p)
                                    #       sigma2: current estimate of measurement error variance (scalar)
                                    #       logl: current estimate of approximated marginal log-likelihood (scalar)
                                    # * see full list of values in 'nlme_classfiles.jl"
    gn_maxiter::Int,                # maximum number of iterations for Gauss-Newton algorithm in Step 1
    gn_err_threshold::FloatType,    # relative convergence threshold for Gauss-Newton algorithm in Step 1
    halving_maxiter::Int,           # maximum number of interations for step-halving procedure in Step 1
    maxiter::Int,                   # maximum number of iterations for MM algorithm
    logl_err_threshold::FloatType   # relative convergence threshold for MM algorithm
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function for estimation of single-level NLME model components described in 
    ##              'Modeling intra-individual inter-trial EEG response variability in Autism
    ##              Spectrum Disorder' Section 3.1, Algorithm 1, including estimation of fixed 
    ##              and random effects, variance components (variance of trial-level random 
    ##              effects and variance of measurement error) and approximated marginal 
    ##              log-likelihood.
    ## Definition:  T: number of trial time points
    ##              R: number of trials
    ##              p: number of trial-level variance components (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Returns:     a tuple of estimated model components
    ##              beta: estimated fixed effects (vector of length p)
    ##              gamma: estimated trial-level random effects (matrix of dimension p*R)
    ##              V: estimated trial-level variance components (diagonal matrix of dimension p*p)
    ##              sigma2: estiamted variance of measurement error (scalar)
    #########################################################################################

    # 'update_mm!': Function that performs MM iteration for single-level NLME, overwrites
    #               model components in NlmeModel type object input, and returns approximated
    #               marginal log-likelihood evaluated at current estimation of model components

    # Perform 1st iteration and save approxiamted marginal log-likelihood into obj
    m.iter = 1
    obj = update_mm_indep!(m, gn_maxiter, gn_err_threshold, halving_maxiter)
    println("iter=1", "obj=", obj)  # print current evaluation of approxiamted marginal log-likelihood
    
    for iter in 2:maxiter # iter: iteration index for MM algorithm, stop when maximum number of 
                          #       iterations is achieved

        # Save estimation of marginal log-likelihood from previous iteration into obj_old
        obj_old = obj

        # Perform one iteration: overwrite model components and iteration index in m, and save 
        # current evaluation of approximated marginal log-likelihood into obj
        obj = update_mm_indep!(m, gn_maxiter, gn_err_threshold, halving_maxiter)
        m.iter = iter

        # Print current evaluation of approximated marginal log-likelihood and relative improvement
        println("iter=", iter, " obj=", obj, " err=", abs(obj - obj_old)/(abs(obj_old) + 1))

        # Check monotonicity: Print a warning message when monotonicity is violated and proceed
        #                     to next iteration, as marginal log-likelihood is approximated
        #                     differently in each iteration
        if obj < obj_old  
            @warn "monotoniciy violated"
        end

        # Check convergence criterion and stop when criterion is met
        abs(obj - obj_old) < logl_err_threshold * (abs(obj_old) + 1) && break

        # Warning about non-convergence
        iter == maxiter && (@warn "maximum iterations reached")
    end

    # Return estimated model components
    return(beta = m.beta, 
           gamma = m.gamma, 
           V = m.V, 
           sigma2 = m.sigma2, 
           logl = m.logl, 
           no_iter = m.iter)
end





function fit_nlme_unstr_mm!(
    m::NlmeUnstrModel{FloatType},   # NlmeUnstrModel type object constructed by NlmeModel_construct
                                    # list of values, including:
                                    #       y_mt: trial-level ERP responses (matrix of dimension T*R)
                                    #       t: trial time grid (vector of length T)
                                    #       beta: current estimates of fixed effects (vector of length p)
                                    #       gamma: current estiamtes of trial-level random effects (matrix of p*R)
                                    #       V: current estimates of trial-level variance components (matrix of dimension p*p)
                                    #       sigma2: current estimate of measurement error variance (scalar)
                                    #       logl: current estimate of approximated marginal log-likelihood (scalar)
                                    # * see full list of values in 'nlme_classfile.jl"
    gn_maxiter::Int,                # maximum number of iterations for Gauss-Newton algorithm in Step 1
    gn_err_threshold::FloatType,    # relative convergence threshold for Gauss-Newton algorithm in Step 1
    halving_maxiter::Int,           # maximum number of interations for step-halving procedure in Step 1
    maxiter::Int,                   # maximum number of iterations for MM algorithm
    logl_err_threshold::FloatType   # relative convergence threshold for MM algorithm # NlmeModel object: trial-level ERP responses and related model parameters
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that performs MM iteration for single-level NLME model with 
    ##              unstructured random effects described in 'Modeling intra-individual inter-
    ##              Spectrum Disorder' Section 3.1, Algorithm 1, overwrites model components
    ##              trial EEG response variability in Autism in NlmeUnstrModel type object input, 
    ##              and returns approxiamted marginal log-likelihood evaluated at current 
    ##              estimation of model components
    ## Definition:  T: number of trial time points
    ##              R: number of trials
    ##              p: dimension of the trial-level covariance matrix (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Overwrites:  values in NlmeModel type object: 
    ##                  beta: estimated fixed effects (vector of length p)
    ##                  gamma: estimated trial-level random effects (matrix of dimension p*R)
    ##                  V: estimated trial-level variance components (matrix of dimension p*p)
    ##                  sigma2: estiamted variance of measurement error (scalar)
    ## Returns:     logl: approximated marginal log-likelihood evaluated at current model 
    ##                    components estimation
    #########################################################################################

    obj = update_mm_unstr!(m, gn_maxiter, gn_err_threshold, halving_maxiter)
    for iter in 0:maxiter
        obj_old = obj
        obj = update_mm_unstr!(m, gn_maxiter, gn_err_threshold, halving_maxiter)
        m.iter = iter
        # print obj
        println("iter=", iter, "obj", obj)
        # check monotonicity
        obj < obj_old  && (@warn "monotoniciy violated") # && break  #
        # check convergence criterion
        abs(obj - obj_old) < logl_err_threshold * (abs(obj_old) + 1) && break
        # warning about non-convergence
        iter == maxiter && (@warn "maximum iterations reached")
    end

end








function update_mm_indep!(
    m::NlmeModel{FloatType},        # NlmeModel type object constructed by NlmeModel_construct
                                    #       list of values, including:
                                    #       y_mt: trial-level ERP responses (matrix of dimension T*R)
                                    #       t: trial time grid (vector of length T)
                                    #       beta: current estimates of fixed effects (vector of length p)
                                    #       gamma: current estiamtes of trial-level random effects (matrix of p*R)
                                    #       V: current estimates of trial-level variance components (diagonal matrix of dimension p*p)
                                    #       sigma2: current estimate of measurement error variance (scalar)
                                    #       logl: current estimate of approximated marginal log-likelihood (scalar)
                                    # * see full list of values in 'nlme_classfiles.jl"
    gn_maxiter::Int,                # maximum number of iterations for Gauss-Newton algorithm in Step 1
    gn_err_threshold::FloatType,    # relative convergence threshold for Gauss-Newton algorithm in Step 1
    halving_maxiter::Int            # maximum number of interations for step-halving procedure in Step 1
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that performs MM iteration for single-level NLME model described
    ##              in 'Modeling intra-individual inter-trial EEG response variability in Autism
    ##              Spectrum Disorder' Section 3.1, Algorithm 1, overwrites model components
    ##              in NlmeModel type object input, and returns approxiamted marginal log-
    ##              likelihood evaluated at current 
    ##              estimation of model components
    ## Definition:  T: number of trial time points
    ##              R: number of trials
    ##              p: number of trial-level variance components (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Overwrites:  values in NlmeModel type object: 
    ##                  beta: estimated fixed effects (vector of length p)
    ##                  gamma: estimated trial-level random effects (matrix of dimension p*R)
    ##                  V: estimated trial-level variance components (diagonal matrix of dimension p*p)
    ##                  sigma2: estiamted variance of measurement error (scalar)
    ## Returns:     logl: approximated marginal log-likelihood evaluated at current model 
    ##                    components estimation
    #########################################################################################

    # Step 1: Update fixed and random effects by solving PNLS via Gauss-Newton algorithm
    T = size(m.t, 1)                
    m.beta_w .= m.beta              # beta_w: estimation of fixed effects from previous iteration
    m.beta_w1 .= m.beta             # beta_w1: estimation of fixed effects from current iteration
    m.gamma_w .= m.gamma            # gamma_w: estimation of random effects from previous iteration
    m.gamma_w1 .= m.gamma           # gamma_w1: estimation of random effects from current iteration

    m.gn_iter = 1                   # gn_iter: iteration index of Gauss-Newton algorithm
    m.gn_err = 10.0                 # gn_err: relative improvement in PNLS objective function from last iteration
    m.pnls_w = 0.0                  # pnls_w: PNLS objective function taking values at previous model components estimations
    m.pnls_w1 = 0.0                 # pnls_w1: PNLS objective function taking values at current model components estimations
    m.V_inv = inv(m.V)              
    m.Delta = (m.V_inv).^(0.5) * sqrt(m.sigma2)     # Delta: precision matrix defined in Gauss-Newton algorithm

    # PNLS objective function to be minimized using Gauss-Newton algorithm
    function pnls_sub(beta, gamma; 
                      y_mt = m.y_mt, R = m.R, Delta = m.Delta, t = m.t)
        pnls = 0.0
        for r in 1:R
            pnls += sum((y_mt[:, r] - mu(t, beta .+ gamma[:, r])).^2) +     # mu: shape function 
                                                                            #     (details see preparation_functions.jl)
                    sum((Delta * gamma[:, r]).^2)
        end
        return pnls
    end

    # Step 1a: Perform iterative Gauss-Newton algorithm
    while m.gn_iter <= gn_maxiter && m.gn_err >= gn_err_threshold
        m.beta_w .= m.beta_w1
        m.gamma_w .= m.gamma_w1

        if m.gn_iter == 1
            m.pnls_w = pnls_sub(m.beta_w, m.gamma_w)
        else
            m.pnls_w = m.pnls_w1
        end

        # Proposed QR decompositions in Step 1
        #       trial_qr_parallel: Function that performs QR decompositions for trial-level 
        #                          design matrices
        #       @sync @distributed: Parallel computation of 'trial_qr_parallel' over trials (Step 1b)
        @sync @distributed for r = 1:m.R
            m.R_1[(5*r-4):(5*r),:], 
            m.S_1[(5*r-4):(5*r),:], 
            m.s_1[(5*r-4):(5*r)],
            m.S_0[(T*r-(T-1)):(T*r),:],
            m.s_0[(T*r-(T-1)):(T*r)] = trial_qr_parallel_indep(m.beta_w, m.gamma_w[:,r], m.Delta, m.y_mt[:,r], m.t, m.phi_r,
                                                               m.M_r, m.w_r, m.K_r, m.R_11_r, m.R_r_right)
        end

        # Calculate LLS estimators using matrices from QR decomposition
        m.beta_w1 .= vec(m.S_0 \ m.s_0)
        for r = 1:m.R
            m.gamma_w1[:,r] .= m.R_1[(5*r-4):(5*r),:] \ (m.s_1[(5*r-4):(5*r)] - m.S_1[(5*r-4):(5*r),:] * m.beta_w1)
        end
 
        # Step halving procedure at end of each iteration of Gauss-Newton algorithm
        m.delta_beta .= m.beta_w1 - m.beta_w
        m.delta_gamma .= m.gamma_w1 - m.gamma_w
        m.halv_iter = 1
        m.halv_err = 20
        while m.halv_iter <= halving_maxiter && m.halv_err >= 0
            m.beta_w1 .= m.beta_w + m.delta_beta
            m.gamma_w1 .= m.gamma_w + m.delta_gamma
            m.pnls_w1 = pnls_sub(m.beta_w1, m.gamma_w1)
            m.halv_err = m.pnls_w1 - m.pnls_w
            m.delta_beta .*= 0.5
            m.delta_gamma .*= 0.5
            m.halv_iter += 1
        end

        # Calculate relative improvement in PNLS objective function
        m.gn_err = abs(m.pnls_w1 - m.pnls_w) / m.pnls_w
        m.gn_iter += 1
    end

    # End of Step 1: Overwrite fixed and random effects in NlmeModel object with estimations
    #                from current iteration
    m.beta .= m.beta_w1
    m.gamma .= m.gamma_w1

    # Taking the absolute values of trial-specific amplitude parameters as their sign do not impact
    # shape function (due to squaring) and it ensures a decrease in PNLS objective function
    m.beta .+= vec(mapslices(mean, m.gamma, dims = 2))
    m.gamma .-= vec(mapslices(mean, m.gamma, dims = 2))
    phi_r = zeros(5)
    for r in 1:m.R
        phi_r .= m.beta + m.gamma[:,r]
        m.gamma[1,r] = abs(phi_r[1]) - m.beta[1]
        m.gamma[3,r] = abs(phi_r[3]) - m.beta[3]
    end
    
    # Step 2: Updating variance components by maximizing minorization function
    #       mm_value_update_parallel: Function that calculates trial-level contributions to numerators
    #                                 and denominators for updating variance components via MM algorithm
    #       @sync @distributed: Parallel computation of 'mm_value_update_parallel' over trials (Step 2b)
    @sync @distributed for r in 1:m.R
        m.mm_update_value[:,r] .= mm_value_update_parallel_indep(m.beta, m.gamma[:,r], m.y_mt[:,r], m.t, 
                                                                 m.sigma2, m.V_inv, m.V_half, m.phi_r, m.M_r, m.omega_r_inv)
    end

    # Calculate numerators and denominators for updating variance components by summing up 
    # trial-specific contributions
    m.mm_update_sum .= sum(m.mm_update_value, dims=2)
    m.logl = m.mm_update_sum[1]
    m.sigma_num = m.mm_update_sum[2]
    m.sigma_den = m.mm_update_sum[3]
    m.V_num .= m.mm_update_sum[4:8]
    m.V_den .= m.mm_update_sum[9:13]

    # Update/overwrite variance components in an element-wise fashion
    m.sigma2 = m.sigma2 * sqrt(m.sigma_num / m.sigma_den)
    m.V .= Diagonal(diag(m.V) .* sqrt.(m.V_num ./ m.V_den))

    # End of Step 2: Return approximated marginal log-likelihood evaluated at current
    #                model components estimation
    return(m.logl)
end




function trial_qr_parallel_indep(
    # current estimation of model components
    beta_w::Vector{FloatType},          # current estimation of fixed effects (vector of length p)
    gamma_w_r::Vector{FloatType},       # current estimation of r-th trial-level random effects (vector of length p)
    Delta::Diagonal{FloatType, Vector{FloatType}},
                                        # precision matrix derived from current estimation of variance components (diagonal matrix of dimension p*p)
    # data
    y_r::Vector{FloatType},             # r-th trial-level ERP response (vector of length T)
    t::Vector{FloatType},               # trial time grid (vector of length T)

    # placeholder for better memory allocation
    phi_r::Vector{FloatType},           # estimated shape parameters for r-th trial (vector of length p)
    M_r::Matrix{FloatType},             # first-order derivative of shape function over shape parameters (matrix of dimension T*p)
    w_r::Vector{FloatType},             # adjusted residual vector (vector of length T)
    K_r::Matrix{FloatType},             # design matrix for r-th trial-level random effects (matrix of dimension (T+p)*p)
    R_1_r::Matrix{FloatType},           # upper triangular matrix of dimension p*p from QR decomposition
    R_r_right::Matrix{FloatType}        # transformed design matrices for fixed effects (Q_0^T S_0) and transformed 
                                        # constant vector (Q_0^T s_0) (Q_0^T[S_0, s_0], matrix of dimension (T+p)*(p+1))
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that performs QR decompositions for r-th trial-specific design 
    ##              matrices described in Section 3.1
    ## Definition:  T: number of trial time points
    ##              p: number of trial-level variance components (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Returns:     a tuple of coefficient matrices and constant vectors for LLS estimator derivation
    ##              R_1_r, S_1_r, s_1_r, S_0_r, s_0_r: notations given in Section 3.1 of paper
    #########################################################################################

    # Calculate current estimation of shape parameters for r-th trial
    phi_r .= beta_w + gamma_w_r

    # Calculate first-order derivative of shape function over shape parameters evaluated at 
    # current estimation of shape parameters
    M_r .= mu1(t, phi_r)        # mu1: first-order derivative of shape function with respective 
                                #      to shape parameters (details see preparation_functions.jl)

    # Calculate adjusted trial-level residual 
    w_r .= y_r .- mu(t, phi_r) .+ M_r * phi_r

    # Derive desigh matrices of r-th trial-level random effects
    K_r .= [M_r; Delta]

    # QR decomposition of trial-level design matrix
    Q_r, R_r = qr(K_r)
    R_1_r .= R_r

    # Design matrix of fixed effects and adjusted trial-level residual (constant) vector, i.e., [L_r, w_r]
    R_r_right .= hcat([M_r; Diagonal(zeros(5))], [w_r; zeros(5)])
    
    # Transformed fixed effects design matrix and transformed constant vector
    R_r_right .= Q_r' * R_r_right

    # Return coefficient matrices and constant vectors for LLS estimator derivation 
    # (notation details given in Section 3.1 of paper)
    return (R_1_r = R_1_r, 
            S_1_r = R_r_right[1:5, 1:5],
            s_1_r = R_r_right[1:5, 6],
            S_0_r = R_r_right[6:end, 1:5],
            s_0_r = R_r_right[6:end, 6])
end




function mm_value_update_parallel_indep(
    # current estimation of model components
    beta::Vector{FloatType},            # current estimation of fixed effects (vector of length p)
    gamma_r::Vector{FloatType},         # current estimation of r-th trial-level random effects (vector of length p)
    y_r::Vector{FloatType},             # ERP responses from r-th trial (vector of length T)
    t::Vector{FloatType},               # trial time grid (vector of length T)
    sigma2_now::FloatType,              # currest estimation of measurement error variance
    V_inv::Diagonal{FloatType, Vector{FloatType}},
                                        # inverse of current estimation of covariance matrix of trial-level random effects
                                        # (V^{-1} diagonal matrix of dimension p*p)
    V_half::Diagonal{FloatType, Vector{FloatType}},
                                        # 'square root' of current estimation of covariance matrix of trial-level random effects
                                        # (V^{1/2} diagonal matrix of dimension p*p)
    
    # placeholder for better memory allocation
    phi_r::Vector{FloatType},           # estimated shape parameters for r-th trial (vector of length p) 
    M_r::Matrix{FloatType},             # first-order derivative of shape function over shape parameters (matrix of dimension T*p)
    omega_r_inv::Matrix{FloatType}      # inverse of approximated trial-level covariance matrix
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that calculates trial-specific contribution to numerators and 
    ##              denominators for updating variance components via MM algorithm for each 
    ##              single trial described in Section 3.1
    ## Definition:  T: number of trial time points
    ##              p: number of trial-level variance components (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Returns:     trial-specific contribution to numerators and denominators for updating 
    ##              variance components
    #########################################################################################

    T = size(t)[1]

    # Calculate current estimation of shape parameters for r-th trial
    phi_r .= beta + gamma_r

    # Calculate first-order derivative of shape function over shape parameters
    M_r .= mu1(t, phi_r)

    # Calculate adjusted trial-level residual
    y_r .+= - mu(t, phi_r) + M_r * gamma_r

    # Calculate inverse of approximated trial-level covariance matrix via Woodbury matrix identity
    omega_r_inv .= 1 / sigma2_now * Diagonal(ones(T)) - 1 / sigma2_now^2 * M_r  *
                    (V_inv + M_r' * M_r/ sigma2_now)^(-1) * M_r'
    
    # Calculate trial-level contribution of approximated marginal log-likelihood
    omega_er = omega_r_inv * y_r
    det_i_vu = det(Diagonal(ones(5)) + V_half * M_r' * M_r * V_half / sigma2_now)
    logl_marg = -0.5 * T * log(2 * π) -
            0.5 * T * log(sigma2_now) - 0.5 * log(det_i_vu) -
            0.5 * omega_er' * y_r

    # Calculate trial-level contribution to numerators and denominators for updating variance components  
    sigma_num = dot(omega_er, omega_er)
    sigma_den = sum(diag(omega_r_inv))
    d_num = zeros(5)
        d_den = zeros(5)
        for j in 1:5
            d_num[j] = sum((omega_er' * M_r[:, j]).^2)
            d_den[j] = M_r[:, j]' * omega_r_inv * M_r[:, j]
        end
    return [logl_marg; sigma_num; sigma_den; d_num; d_den]
end



function start_nlme_indep!(
    m::NlmeModel{FloatType},        # NlmeModel type object constructed by NlmeModel_construct
                                    #       list of values, including:
                                    #       y_mt: trial-level ERP responses (matrix of dimension T*R)
                                    #       t: trial time grid (vector of length T)
                                    #       beta: current estimates of fixed effects (vector of length p)
                                    #       gamma: current estiamtes of trial-level random effects (matrix of p*R)
                                    #       V: current estimates of trial-level variance components (diagonal matrix of dimension p*p)
                                    #       sigma2: current estimate of measurement error variance (scalar)
                                    # * see full list of values in 'nlme_classfiles.jl"
    peak_range::Vector{FloatType},  # range of interval to search for peak-shaped ERP component (vector of length 2)
    dip_range::Vector{FloatType}    # range of interval to search for dip-shaped ERP component (vector of length 2)
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that calculates starting values of model components (fixed
    ##              effects and variance components) via detecting ERP components on each trial-
    ##              level ERP responses and overwrites estimations in `NlmeModel` object
    ## Definition:  T: number of trial time points
    ##              R: number of trials
    ##              p: number of trial-level variance components (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Overwrites:  beta, V, sigma2 in input `m`
    #########################################################################################

    # Detect ERP components (P100: peak, N75: dip) on each trial-level ERP response
    #   find_erp: Function that detects peak- or dip-shaped component from a single curve
    #             based on given seracing interval and returns its latency and amplitude
    @inbounds for r in 1:m.R
        y_now =  m.y_mt[:,r]
        n1 = find_erp(m.t, y_now, dip_range, 1.0, false)
        p1 = find_erp(m.t, y_now, peak_range, 1.0, true)
        a = sqrt(abs(p1.val - n1.val)/2)
        m.gamma[1,r] = a
        m.gamma[3,r] = a
        m.gamma[2,r] = n1.arg
        m.gamma[4,r] = p1.arg
        m.gamma[5,r] = mean(y_now)
    end
    m.beta .= vec(mapslices(mean, m.gamma, dims = 2))
    m.gamma .-= m.beta
    m.V .= Diagonal([vec(mapslices(var, m.gamma[1:4,:], dims = 2));1]) 
                    # constant 10 is set to avoid potential explode in trial-level varaince of mean magnitude
    m.gamma = zeros(5, m.R)
    m.sigma2 = 10.0   
    m.V_half .= (m.V).^(0.5)
    m.V_inv .= inv(m.V)
    m.Delta .= (m.V_inv).^(0.5) * sqrt(m.sigma2)
end





function update_mm_unstr!(
    m::NlmeUnstrModel{FloatType},   # NlmeUnstrModel type object constructed by NlmeUnstrModel_construct
                                    #       list of values, including:
                                    #       y_mt: trial-level ERP responses (matrix of dimension T*R)
                                    #       t: trial time grid (vector of length T)
                                    #       beta: current estimates of fixed effects (vector of length p)
                                    #       gamma: current estiamtes of trial-level random effects (matrix of p*R)
                                    #       V: current estimates of trial-level variance components (diagonal matrix of dimension p*p)
                                    #       sigma2: current estimate of measurement error variance (scalar)
                                    #       logl: current estimate of approximated marginal log-likelihood (scalar)
                                    # * see full list of values in 'nlme_classfiles.jl"
    gn_maxiter::Int,                # maximum number of iterations for Gauss-Newton algorithm in Step 1
    gn_err_threshold::FloatType,    # relative convergence threshold for Gauss-Newton algorithm in Step 1
    halving_maxiter::Int            # maximum number of interations for step-halving procedure in Step 1
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that performs MM iteration for single-level NLME model with 
    ##              unstructured random effects described in 'Modeling intra-individual inter-
    ##              trial EEG response variability in Autism Spectrum Disorder' Section 3.1, 
    ##              Algorithm 1, overwrites model components in NlmeUnstrModel type object 
    ##              input, and returns approxiamted marginal log-likelihood evaluated at current 
    ##              estimation of model components
    ## Definition:  T: number of trial time points
    ##              R: number of trials
    ##              p: dimension of trial-level covariance matrix (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Overwrites:  values in NlmeUnstrModel type object: 
    ##                  beta: estimated fixed effects (vector of length p)
    ##                  gamma: estimated trial-level random effects (matrix of dimension p*R)
    ##                  V: estimated trial-level covariance matrix (matrix of dimension p*p)
    ##                  sigma2: estiamted variance of measurement error (scalar)
    ## Returns:     logl: approximated marginal log-likelihood evaluated at current model 
    ##                    components estimation
    #########################################################################################
    
    # Step 1: Update fixed and random effects by solving PNLS via Gauss-Newton algorithm
    TT = size(m.t, 1)
    m.beta_w .= m.beta              # beta_w: estimation of fixed effects from previous iteration
    m.beta_w1 .= m.beta             # beta_w1: estimation of fixed effects from current iteration
    m.gamma_w .= m.gamma            # gamma_w: estimation of random effects from previous iteration
    m.gamma_w1 .= m.gamma           # gamma_w1: estimation of random effects from current iteration

    m.gn_iter = 1                   # gn_iter: iteration index of Gauss-Newton algorithm
    m.gn_err = 10.0                 # gn_err: relative improvement in PNLS objective function from last iteration
    m.pnls_w = 0.0                  # pnls_w: PNLS objective function taking values at previous model components estimations
    m.pnls_w1 = 0.0                 # pnls_w1: PNLS objective function taking values at current model components estimations
    V_eigen = eigen(m.V)
    m.V_half .= V_eigen.vectors * Diagonal(sqrt.(V_eigen.values)) * V_eigen.vectors'
    m.V_inv .= V_eigen.vectors * Diagonal(1 ./ V_eigen.values) * V_eigen.vectors'
    m.Delta .= sqrt(m.sigma2) * V_eigen.vectors * Diagonal(1 ./ sqrt.(V_eigen.values)) * V_eigen.vectors'

    # PNLS objective function to be minimized using Gauss-Newton algorithm
    function pnls_sub(beta, gamma; y_mt = m.y_mt, R = m.R, Delta = m.Delta, t = m.t)
        pnls = 0.0
        for r in 1:R
            pnls += sum((y_mt[:, r] - mu(t, beta .+ gamma[:, r])).^2) +     # mu: shape function 
                                                                            #     (details see preparation_functions.jl)
                    sum((Delta * gamma[:, r]).^2)
        end
        return pnls
    end

    # Step 1: Updating fixed and random effects by solving PNLS via Gauss-Newton algorithm
    while m.gn_iter <= gn_maxiter && m.gn_err >= gn_err_threshold
        m.beta_w .= m.beta_w1
        m.gamma_w .= m.gamma_w

        if m.gn_iter == 1
            m.pnls_w = pnls_sub(m.beta_w, m.gamma_w)
        else
            m.pnls_w = m.pnls_w1
        end

        # Proposed QR decompositions in Step 1
        #       trial_qr_parallel_unstr: Function that performs QR decompositions for trial-level 
        #                          design matrices
        #       @sync @distributed: Parallel computation of 'trial_qr_parallel_unstr' over trials (Step 1b)
        @sync @distributed for r = 1:m.R ## parallel computation over trials
            m.R_1[(5*r-4):(5*r),:], 
            m.S_1[(5*r-4):(5*r),:], 
            m.s_1[(5*r-4):(5*r)],
            m.S_0[(TT*r-(TT-1)):(TT*r),:],
            m.s_0[(TT*r-(TT-1)):(TT*r)] = trial_qr_parallel_unstr(m.beta_w, m.gamma_w[:,r], m.Delta, m.y_mt[:,r], m.t, m.phi_r,
                                                        m.M_r, m.w_r, m.K_r, m.R_11_r, m.R_r_right)
        end

        # Calculate LLS estimators using matrices from QR decomposition
        m.beta_w1 .= vec(m.S_0 \ m.s_0)
        for r = 1:m.R
            m.gamma_w1[:,r] .= m.R_1[(5*r-4):(5*r),:] \ (m.s_1[(5*r-4):(5*r)] - m.S_1[(5*r-4):(5*r),:] * m.beta_w1)
        end 

        # Step halving procedure at end of each iteration of Gauss-Newton algorithm
        m.delta_beta .= m.beta_w1 - m.beta_w
        m.delta_gamma .= m.gamma_w1 - m.gamma_w
        m.halv_iter = 1
        m.halv_err = 20
        while m.halv_iter <= halving_maxiter && m.halv_err >= 0
            m.beta_w1 .= m.beta_w + m.delta_beta
            m.gamma_w1 .= m.gamma_w + m.delta_gamma
            m.pnls_w1 = pnls_sub(m.beta_w1, m.gamma_w1)
            m.halv_err = m.pnls_w1 - m.pnls_w
            m.delta_beta .*= 0.5
            m.delta_gamma .*= 0.5
            m.halv_iter += 1
        end

        # Calculate relative improvement in PNLS objective function
        m.gn_err = abs(m.pnls_w1 - m.pnls_w) / m.pnls_w
        m.gn_iter += 1

    end

    # End of Step 1: Overwrite fixed and random effects in NlmeModel object with estimations
    #                from current iteration
    m.beta .= m.beta_w1
    m.gamma .= m.gamma_w1

    # Taking the absolute values of trial-specific amplitude parameters as their sign do not impact
    # shape function (due to squaring) and it ensures a decrease in PNLS objective function
    m.beta .+= vec(mapslices(mean, m.gamma, dims = 2))
    m.gamma .-= vec(mapslices(mean, m.gamma, dims = 2))
    phi_r = zeros(5)
    for r in 1:m.R
        phi_r .= m.beta + m.gamma[:,r]
        m.gamma[1,r] = abs(phi_r[1]) - m.beta[1]
        m.gamma[3,r] = abs(phi_r[3]) - m.beta[3]
    end
    

    # Step 2: Updating variance components by maximizing minorization function
    #       mm_value_update_parallel_unstr: Function that calculates trial-level contributions to numerators
    #                                 and denominators for updating variance components via MM algorithm
    #       @sync @distributed: Parallel computation of 'mm_value_update_parallel_unstr' over trials (Step 2b)
    @sync @distributed for r in 1:m.R
        m.mm_update_value[:,r],
        m.m_omega_m_series[(5*r-4):(5*r),:],
        m.ss_series[(5*r-4):(5*r),:] = mm_value_update_parallel_unstr(m.beta, m.gamma[:,r], m.y_mt[:,r], m.t, 
                                                        m.sigma2, m.V, m.V_inv, m.V_half, m.phi_r, m.M_r, m.omega_r_inv)
    end

    # Calculate numerators and denominators for updating variance components by summing up 
    # trial-specific contributions
    m.mm_update_sum .= sum(m.mm_update_value, dims=2)
    m.m_omega_m_sum .= m.m_omega_m_series[1:5,:]
    m.ss_sum .= m.ss_series[1:5, :]
    for r in 2:m.R
        m.m_omega_m_sum .+= m.m_omega_m_series[(5*r-4):(5*r),:]
        m.ss_sum .+= m.ss_series[(5*r-4):(5*r),:]
    end
    m.m_omega_m_sum .= 0.5 * (m.m_omega_m_sum + m.m_omega_m_sum')
    m.ss_sum .= 0.5 * (m.ss_sum + m.ss_sum')

    # Update/overwrite variance for residual error in an element-wise fashion
    m.logl = m.mm_update_sum[1]
    m.sigma_num = m.mm_update_sum[2]
    m.sigma_den = m.mm_update_sum[3]
    m.sigma2 = m.sigma2 * sqrt(m.sigma_num / m.sigma_den)
    
    # Update/overwrite trial-level covariance matrix via explicit matrix formula
    L_c = cholesky(m.m_omega_m_sum).L
    L_c_inv = inv(L_c)
    L_ss_L = L_c' * m.ss_sum * L_c
    L_ss_L_eigen = eigen(L_ss_L)
    L_ss_L_half = L_ss_L_eigen.vectors * Diagonal(sqrt.(L_ss_L_eigen.values)) * L_ss_L_eigen.vectors'
    m.V .= L_c_inv' * L_ss_L_half * L_c_inv
    
    # End of Step 2: Return approximated marginal log-likelihood evaluated at current
    #                model components estimation
    return(m.logl)
end




function trial_qr_parallel_unstr(
    # current estimation of model components
    beta_w::Vector{FloatType},          # current estimation of fixed effects (vector of length p)
    gamma_w_r::Vector{FloatType},       # current estimation of r-th trial-level random effects (vector of length p)
    Delta::Matrix{FloatType},           # precision matrix derived from current estimation of covariance matrix
                                        #   (matrix of dimension p*p)
    y_r::Vector{FloatType},
    t::Vector{FloatType},

    # placeholder for better memory allocation
    phi_r::Vector{FloatType},           # estimated shape parameters for r-th trial (vector of length p)
    M_r::Matrix{FloatType},             # first-order derivative of shape function over shape parameters (matrix of dimension T*p)
    w_r::Vector{FloatType},             # adjusted residual vector (vector of length T)
    K_r::Matrix{FloatType},             # design matrix for r-th trial-level random effects (matrix of dimension (T+p)*p)
    R_1_r::Matrix{FloatType},           # upper triangular matrix of dimension p*p from QR decomposition
    R_r_right::Matrix{FloatType}        # transformed design matrices for fixed effects (Q_0^T S_0) and transformed 
                                        # constant vector (Q_0^T s_0) (Q_0^T[S_0, s_0], matrix of dimension (T+p)*(p+1))
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that performs QR decompositions for r-th trial-specific design 
    ##              matrices described in Section 3.1
    ## Definition:  T: number of trial time points
    ##              p: number of trial-level variance components (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Returns:     a tuple of coefficient matrices and constant vectors for LLS estimator derivation
    ##              R_1_r, S_1_r, s_1_r, S_0_r, s_0_r: notations given in Section 3.1 of paper
    #########################################################################################

    # Calculate current estimation of shape parameters for r-th trial
    phi_r .= beta_w + gamma_w_r

    # Calculate first-order derivative of shape function over shape parameters evaluated at 
    # current estimation of shape parameters
    M_r .= mu1(t, phi_r)        # mu1: first-order derivative of shape function with respective 
                                #      to shape parameters (details see preparation_functions.jl)

    # Calculate adjusted trial-level residual 
    w_r .= y_r .- mu(t, phi_r) .+ M_r * phi_r

    # Derive desigh matrices of r-th trial-level random effects
    K_r .= [M_r; Delta]

    # QR decomposition of trial-level design matrix
    Q_r, R_r = qr(K_r)
    R_1_r .= R_r

    # Design matrix of fixed effects and adjusted trial-level residual (constant) vector, i.e., [L_r, w_r]
    R_r_right .= hcat([M_r; Diagonal(zeros(5))], [w_r; zeros(5)])
    
    # Transformed fixed effects design matrix and transformed constant vector
    R_r_right .= Q_r' * R_r_right

    # Return coefficient matrices and constant vectors for LLS estimator derivation 
    # (notation details given in Section 3.1 of paper)
    return (R_1_r = R_1_r, 
            S_1_r = R_r_right[1:5, 1:5],
            s_1_r = R_r_right[1:5, 6],
            S_0_r = R_r_right[6:end, 1:5],
            s_0_r = R_r_right[6:end, 6])
end





function mm_value_update_parallel_unstr(
    # current estimation of model components
    beta::Vector{FloatType},            # current estimation of fixed effects (vector of length p)
    gamma_r::Vector{FloatType},         # current estimation of r-th trial-level random effects (vector of length p)
    y_r::Vector{FloatType},             # ERP responses from r-th trial (vector of length T)
    t::Vector{FloatType},               # trial time grid (vector of length T)
    sigma2_now::FloatType,              # currest estimation of measurement error variance
    V::Matrix{FloatType},               # current estimation of covariance matrix of trial-level random effects
    V_inv::Matrix{FloatType},           # inverse of current estimation of covariance matrix of trial-level random effects
                                        # (V^{-1} diagonal matrix of dimension p*p)
    V_half::Matrix{FloatType},          # 'square root' of current estimation of covariance matrix of trial-level random effects
                                        # (V^{1/2} diagonal matrix of dimension p*p)
    
    # placeholder for better memory allocation
    phi_r::Vector{FloatType},           # estimated shape parameters for r-th trial (vector of length p) 
    M_r::Matrix{FloatType},             # first-order derivative of shape function over shape parameters (matrix of dimension T*p)
    omega_r_inv::Matrix{FloatType}      # inverse of approximated trial-level covariance matrix
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that calculates trial-specific contribution to scalars and
    ##              matrices for updating variance components via MM algorithm in each 
    ##              single trial described in Section 3.1
    ## Definition:  T: number of trial time points
    ##              p: dimension of trial-level covariance matrix (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Returns:     trial-specific contribution to scalars and matrices for updating 
    ##              variance components
    #########################################################################################

    T = size(t)[1]

    # Calculate current estimation of shape parameters for r-th trial
    phi_r .= beta + gamma_r

    # Calculate first-order derivative of shape function over shape parameters
    M_r .= mu1(t, phi_r)

    # Calculate adjusted trial-level residual
    y_r .+= - mu(t, phi_r) + M_r * gamma_r

    # Calculate inverse of approximated trial-level covariance matrix via Woodbury matrix identity
    omega_r_inv .= 1 / sigma2_now * Diagonal(ones(T)) - 1 / sigma2_now^2 * M_r  *
                    (V_inv + M_r' * M_r/ sigma2_now)^(-1) * M_r'
    
    # Calculate trial-level contribution of approximated marginal log-likelihood
    omega_er = omega_r_inv * y_r
    s_r = V * M_r' * omega_er
    det_i_vu = det(Diagonal(ones(5)) + V_half * M_r' * M_r * V_half / sigma2_now)
    logl_marg = -0.5 * T * log(2 * π) -
            0.5 * T * log(sigma2_now) - 0.5 * log(det_i_vu) -
            0.5 * omega_er' * y_r
    
    # Calculate trial-level contribution to numerators and denominators for updating variance components  
    sigma_num = dot(omega_er, omega_er)
    sigma_den = sum(diag(omega_r_inv))
    m_omega_m_r = M_r' * omega_r_inv * M_r
    ss_r = s_r * s_r'
    
    return (mm_value = [logl_marg; sigma_num; sigma_den],
            m_omega_m_r = m_omega_m_r,
            ss_r = ss_r)
end




# Function - init_est!(): update the starting values of the NlmeModel type object
function start_nlme_unstr!(
    m::NlmeModel{FloatType},        # NlmeUnstrModel type object constructed by NlmeModel_construct
                                    #       list of values, including:
                                    #       y_mt: trial-level ERP responses (matrix of dimension T*R)
                                    #       t: trial time grid (vector of length T)
                                    #       beta: current estimates of fixed effects (vector of length p)
                                    #       gamma: current estimates of trial-level random effects (matrix of p*R)
                                    #       V: current estimates of trial-level variance components (diagonal matrix of dimension p*p)
                                    #       sigma2: current estimate of measurement error variance (scalar)
                                    # * see the full list of values in 'nlme_classfile.jl"
    peak_range::Vector{FloatType},  # range of interval to search for peak-shaped ERP component (vector of length 2)
    dip_range::Vector{FloatType}    # range of interval to search for dip-shaped ERP component (vector of length 2)
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that calculates starting values of model components (fixed
    ##              effects and variance components) via detecting ERP components on each trial-
    ##              level ERP responses and overwrites estimations in `NlmeModel` object
    ## Definition:  T: number of trial time points
    ##              R: number of trials
    ##              p: dimension of trial-level covariance matrix (equals to number of shape 
    ##                 parameters in shape function)
    ## Args:        see above
    ## Overwrites:  beta, V, sigma2 in input `m`
    #########################################################################################

    # Detect ERP components (P100: peak, N75: dip) on each trial-level ERP response
    #   find_erp: Function that detects peak- or dip-shaped component from a single curve
    #             based on given seracing interval and returns its latency and amplitude
    @inbounds for r in 1:m.R
    @inbounds for r in 1:m.R
        y_now =  m.y_mt[:,r]
        n1 = find_erp(m.t, y_now, dip_range, 1.0, false)
        p1 = find_erp(m.t, y_now, peak_range, 1.0, true)
        a = sqrt(abs(p1.val - n1.val)/2)
        m.gamma[1,r] = a
        m.gamma[3,r] = a
        m.gamma[2,r] = n1.arg
        m.gamma[4,r] = p1.arg
        m.gamma[5,r] = mean(y_now)
    end
    m.beta .= vec(mapslices(mean, m.gamma, dims = 2))
    m.gamma .-= m.beta
    m.sigma2 = 10.0
    v_diag = [vec(mapslices(var, m.gamma[1:4,:], dims = 2));1.0]
                # constant 1.0 is set to avoid potential explode in trial-level varaince of mean magnitude
    m.V .= zeros(T, 5, 5)
    m.V_half .= zeros(T, 5, 5)
    m.V_inv .= zeros(T, 5, 5)
    m.Delta .= zeros(T, 5, 5)
    for i in 1:5
        m.V[i,i] = v_diag[i]
        m.V_half[i,i] = sqrt(v_diag[i])
        m.V_inv[i,i] = 1/v_diag[i]
        m.Delta[i,i] = sqrt(m.sigma2/v_diag[i])
    end
    m.gamma .= zeros(T, 5, m.R)
end
