###########################################################################################
## Description: Functions for fitting multi-level NLME model via MM algorithm described
##              in 'Modeling intra-individual inter-trial EEG response variability in Autism 
##              Spectrum Disorder'.
###########################################################################################
## Functions included:
## Main function:
##    1. fit_mnlme_indep_mm!: Function that fits multi-level NLME model with independent random
##                            effects via MM algorithm
##    2. fit_mnlme_unstr_mm!: Function that fits multi-level NLME model with unstructured 
##                            random effects via MM algorithm
## Supporting functions used by main function:
##    (For independent models)    
##    1.1 update_mnlme_indep_mm!: Function that performs MM iteration
##    1.2 subject_qr_parallel_indep: Function that performs QR decomposition in Step1 of each iteration
##    1.3 mnlme_indep_mm_update_parallel: Function that calculates components for updating variance 
##                                        components in Step 2 of each iteration
##    1.4 start_mnlme_indep!: Function that calculates starting values of model components from
##                            single-level model fits
##    (For unstructured models)    
##    2.1 update_mnlme_unstr_mm!: Function that performs MM iteration
##    2.2 subject_qr_parallel_unstr: Function that performs QR decomposition in Step1 of each 
##                                   iteration
##    2.3 mnlme_unstr_mm_update_parallel: Function that calculates components for updating
##                                        variance components in Step 2 of each iteration
##    2.4 start_mnlme_unstr!: Function that calculates starting values of model components 
##                            from single-level model fits
###########################################################################################





function fit_mnlme_indep_mm!(
    m::MnlmeModel{FloatType},       # MnlmeModel type object constructed by MnlmeModel_construct
                                    # list of values, including:
                                    #       y_mt_array: trial-level ERP responses for each subject
                                    #                   (array of matrices of dimension T*R_i, i=1,...,n)
                                    #       t: trial time grid (vector of length T)
                                    #       beta: current estimates of fixed effects (vector of length p)
                                    #       alpha: current estimation of subject-level random effects (matrix of p*n)
                                    #       gamma: current estiamtes of trial-level random effects for each subject 
                                    #              (array of matrices of dimension p*R_i, i=1,...,n)
                                    #       U: current estimates of subject-level variance components (diagonal matrix of dimension p*p)
                                    #       V: current estimates of trial-level variance components (diagonal matrix of dimension p*p)
                                    #       sigma2: current estimate of measurement error variance (scalar)
                                    #       logl: current estimate of approximated marginal log-likelihood (scalar)
                                    # * see full list of values in 'nlme_classfiles.jl"
    gn_maxiter::Int,                # maximum number of iterations for Gauss-Newton algorithm
    gn_err_threshold::FloatType,    # relative convergence threshold for Gauss-Newton algorithm
    halving_maxiter::Int,           # maximum times of step-halving
    maxiter::Int,                   # maximum number of iterations for iterative MM algorithm for multi-level NLME
    logl_err_threshold::FloatType   # relative convergence threshold for MM algorithm for multi-level NLME
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function for estimation of multi-level NLME model components described in 
    ##              'Modeling intra-individual inter-trial EEG response variability in Autism
    ##              Spectrum Disorder' Section 3.2, Algorithm 2, including estimation of fixed 
    ##              and random effects, variance components (variance of subject- and trial-
    ##              level random effects and variance of measurement error) and approximated 
    ##              marginal log-likelihood.
    ## Definition:  T: number of trial time points
    ##              n: number of subjects
    ##              R_i: number of trials for subject i, i=1,...,n
    ##              p: number of shape parameters (equals to number of variance components on 
    ##                 each level)
    ## Args:        see above
    ## Returns:     a tuple of estimated model components
    ##              beta: estimated fixed effects (vector of length p)
    ##              alpha: estimated subject-level random effects (matrix of dimension p*n)
    ##              gamma: estimated trial-level random effects for each subject
    ##                     (array of matrices of dimension p*R_i, i=1,...,n)
    ##              U: estimated subject-level variance components (diagonal matrix of dimension p*p)
    ##              V: estimated trial-level variance components (diagonal matrix of dimension p*p)
    ##              sigma2: estiamted variance of measurement error (scalar)
    #########################################################################################

    # 'update_mnlme_indep_mm!': Function that performs MM iteration for multi-level NLME, overwrites
    #                           model components in MnlmeModel type object input, and returns 
    #                           approximated marginal log-likelihood evaluated at current estimation 
    #                           of model components

    # Perform 1st iteration and save approxiamted marginal log-likelihood into obj
    m.iter = 1
    obj = update_mnlme_indep_mm!(m, gn_maxiter, gn_err_threshold, halving_maxiter)
    println("iter=1", "obj=", obj)  # print current evaluation of approxiamted marginal log-likelihood

    for iter = 2:maxiter    # iter: iteration index for MM algorithm, stop when maximum number of 
                            #       iterations is achieved
        
        # Save estimation of marginal log-likelihood from previous iteration into obj_old
        obj_old = obj

        # Perform one iteration: overwrite model components and iteration index in m, and save 
        # current evaluation of approximated marginal log-likelihood into obj
        obj = update_mnlme_indep_mm!(m, gn_maxiter, gn_err_threshold, halving_maxiter)
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
           alpha = m.alpha,
           gamma = m.gamma,
           U = m.U,
           V = m.V,
           sigma2 = m.sigma2,
           logl = m.logl,
           no_iter = m.iter)
end





function fit_mnlme_unstr_mm!(
    m::MnlmeUnstrModel{FloatType},  # MnlmeUnstrModel type object constructed by MnlmeUnstrModel_construct
                                    # list of values, including:
                                    #       y_mt_array: trial-level ERP responses for each subject
                                    #                   (array of matrices of dimension T*R_i, i=1,...,n)
                                    #       t: trial time grid (vector of length T)
                                    #       beta: current estimates of fixed effects (vector of length p)
                                    #       alpha: current estimation of subject-level random effects (matrix of p*n)
                                    #       gamma: current estiamtes of trial-level random effects for each subject 
                                    #              (array of matrices of dimension p*R_i, i=1,...,n)
                                    #       U: current estimates of subject-level covariance matrix (matrix of dimension p*p)
                                    #       V: current estimates of trial-level covariance matrix (matrix of dimension p*p)
                                    #       sigma2: current estimate of measurement error variance (scalar)
                                    #       logl: current estimate of approximated marginal log-likelihood (scalar)
                                    # * see full list of values in 'nlme_classfiles.jl"
    gn_maxiter::Int,                # maximum number of iterations for Gauss-Newton algorithm
    gn_err_threshold::FloatType,    # relative convergence threshold for Gauss-Newton algorithm
    halving_maxiter::Int,           # maximum times of step-halving
    maxiter::Int,                   # maximum number of iterations for iterative MM algorithm for multi-level NLME
    logl_err_threshold::FloatType   # relative convergence threshold for MM algorithm for multi-level NLME
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function for estimation of multi-level NLME model components described in 
    ##              'Modeling intra-individual inter-trial EEG response variability in Autism
    ##              Spectrum Disorder' Section 3.2, Algorithm 2, including estimation of fixed 
    ##              and random effects, variance components (covariance matrices of subject- 
    ##              and trial-level random effects and variance of measurement error) and 
    ##              approximated marginal log-likelihood.
    ## Definition:  T: number of trial time points
    ##              n: number of subjects
    ##              R_i: number of trials for subject i, i=1,...,n
    ##              p: number of shape parameters (equals to number of variance components on 
    ##                 each level)
    ## Args:        see above
    ## Returns:     a tuple of estimated model components
    ##              beta: estimated fixed effects (vector of length p)
    ##              alpha: estimated subject-level random effects (matrix of dimension p*n)
    ##              gamma: estimated trial-level random effects for each subject
    ##                     (array of matrices of dimension p*R_i, i=1,...,n)
    ##              U: estimated subject-level covariance matrix (matrix of dimension p*p)
    ##              V: estimated trial-level covariance matrix (matrix of dimension p*p)
    ##              sigma2: estiamted variance of measurement error (scalar)
    #########################################################################################

    # 'update_mnlme_unstr_mm!': Function that performs MM iteration for multi-level NLME, overwrites
    #                     model components in MnlmeModel type object input, and returns 
    #                     approximated marginal log-likelihood evaluated at current estimation 
    #                     of model components

    # Perform 1st iteration and save approxiamted marginal log-likelihood into obj
    m.iter = 1
    obj = update_mnlme_unstr_mm!(m, gn_maxiter, gn_err_threshold, halving_maxiter)
    println("iter=1", "obj=", obj)  # print current evaluation of approxiamted marginal log-likelihood

    for iter = 2:maxiter    # iter: iteration index for MM algorithm, stop when maximum number of 
                            #       iterations is achieved
        
        # Save estimation of marginal log-likelihood from previous iteration into obj_old
        obj_old = obj

        # Perform one iteration: overwrite model components and iteration index in m, and save 
        # current evaluation of approximated marginal log-likelihood into obj
        obj = update_mnlme_mm!(m, gn_maxiter, gn_err_threshold, halving_maxiter)
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
           alpha = m.alpha,
           gamma = m.gamma,
           U = m.U,
           V = m.V,
           sigma2 = m.sigma2,
           logl = m.logl,
           no_iter = m.iter)
end







function update_mnlme_indep_mm!(
    m::MnlmeModel{FloatType},       # MnlmeModel type object constructed by MnlmeModel_construct
                                    # list of values, including:
                                    #       y_mt_array: trial-level ERP responses for each subject
                                    #                   (array of matrices of dimension T*R_i, i=1,...,n)
                                    #       t: trial time grid (vector of length T)
                                    #       beta: current estimates of fixed effects (vector of length p)
                                    #       alpha: current estimation of subject-level random effects (matrix of p*n)
                                    #       gamma: current estiamtes of trial-level random effects for each subject 
                                    #              (array of matrices of dimension p*R_i, i=1,...,n)
                                    #       U: current estimates of subject-level variance components (diagonal matrix of dimension p*p)
                                    #       V: current estimates of trial-level variance components (diagonal matrix of dimension p*p)
                                    #       sigma2: current estimate of measurement error variance (scalar)
                                    #       logl: current estimate of approximated marginal log-likelihood (scalar)
                                    # * see full list of values in 'nlme_classfiles.jl"
    gn_maxiter::Int,                # maximum number of iterations for Gauss-Newton algorithm
    gn_err_threshold::FloatType,    # relative convergence threshold for Gauss-Newton algorithm
    halving_maxiter::Int            # maximum times of step-halving
) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that performs MM iteration for multi-level NLME model described
    ##              in 'Modeling intra-individual inter-trial EEG response variability in Autism
    ##              Spectrum Disorder' Section 3.2, Algorithm 2, overwrites model components
    ##              in NlmeModel type object input, and returns approxiamted marginal log-
    ##              likelihood evaluated at current 
    ##              estimation of model components
    ## Definition:  T: number of trial time points
    ##              n: number of subjects
    ##              R_i: number of trials for subject i, i=1,...,n
    ##              p: number of shape parameters (equals to number of variance components on 
    ##                 each level)
    ## Args:        see above
    ## Overwrites:  values in NlmeModel type object: 
    ##                  beta: estimated fixed effects (vector of length p)
    ##                  alpha: estimated subject-level random effects (matrix of dimension p*n)
    ##                  gamma: estimated trial-level random effects for each subject
    ##                         (array of matrices of dimension p*R_i, i=1,...,n)
    ##                  U: estimated subject-level variance components (diagonal matrix of dimension p*p)
    ##                  V: estimated trial-level variance components (diagonal matrix of dimension p*p)
    ##                  sigma2: estiamted variance of measurement error (scalar)
    ## Returns:     logl: approximated marginal log-likelihood evaluated at current model 
    ##                    components estimation
    #########################################################################################

    # Preparation for parallel computation: schedule computational nodes for subject-level jobs
    # Create storage spaces for results from parallel computational nodes
    futures = Vector{Future}(undef, m.n)
    # Assigns each subset to a worker
    wks_schedule = Vector{Int}(undef, m.n)
    wk_list = workers()
    wk = 1
    wk_max = size(wk_list,1)
    @inbounds for i = 1:m.n
        wks_schedule[i] = wk_list[wk]
        wk == wk_max ? wk = 1 : wk += 1
    end
    flush(stdout)
    print("wks_schedule = ", wks_schedule, "\n")

    # Step 1: Update fixed and random effects by solving PNLS via Gauss-Newton Algorithm
    m.beta_w .= m.beta              # beta_w: estimation of fixed effects from previous iteration
    m.beta_w1 .= m.beta             # beta_w1: estimation of fixed effects from current iteration
    m.alpha_w .= m.alpha            # alpha_w: estimation of subject-level random effects from previous iteration
    m.alpha_w1 .= m.alpha           # alpha_w1: estimation of subject-level random effects from current iteration
    for i=1:m.n
        m.gamma_w[i] .= m.gamma[i]  # gamma_w: estimation of trial-level random effects from previous iteration
        m.gamma_w1[i] .= m.gamma[i] # gamma_w1: estimation of trial-level random effects from current iteration
    end

    m.gn_iter = 1                   # gn_iter: iteration index of Gauss-Newton algorithm
    m.gn_err = 10.0                 # gn_err: relative improvement in PNLS objective function from last iteration
    m.pnls_w  = 0.0                 # pnls_w: PNLS objective function taking values at previous model components estimation
    m.pnls_w1 = 0.0                 # pnls_w1: PNLS objective function taking values at current model components estimation
    m.U_inv .= inv(m.U)
    m.U_half .= m.U.^0.5
    m.Delta_U .= (m.U_inv).^0.5 * sqrt(m.sigma2)    # Delta_U: subject-level precision matrix defined in Gauss-Newton algorithm
    m.V_inv .= inv(m.V)
    m.V_half .= m.V.^0.5
    m.Delta_V .= (m.V_inv).^0.5 * sqrt(m.sigma2)    # Delta_V: trial-level precision matrix defined in Gauss-Newton algorithm

    # PNLS objective function to be minimized using Gauss-Newton algorithm
    function pnls_all(beta, alpha, gamma;
                        y_mt_array = m.y_mt_array,
                        n = m.n,
                        t = m.t,
                        R_array = m.R_array,
                        Delta_u = m.Delta_U,
                        Delta_v = m.Delta_V)
        pnls = 0.0
        @inbounds for i = 1:n
            R_i = R_array[i]
            @inbounds for r = 1:R_i
                pnls += sum((y_mt_array[i][:,r] - mu(t, beta .+ alpha[:,i] .+ gamma[i][:,r])).^2) +
                        sum((Delta_v * gamma[i][:,r]).^2)       # mu: shape function 
                                                                #     (details see preparation_functions.jl)
            end
            pnls += sum((Delta_u * alpha[:,i]).^2)
        end
        return pnls
    end

    # Step 1a: Perform iterative Gauss-Newton algorithm
    while m.gn_iter <= gn_maxiter && m.gn_err >= gn_err_threshold
        m.beta_w .= m.beta_w1
        m.alpha_w .= m.alpha_w1
        @inbounds for i=1:m.n
            m.gamma_w[i] .= m.gamma_w1[i]
        end

        if m.gn_iter == 1
            m.pnls_w = pnls_all(m.beta_w, m.alpha_w, m.gamma_w)
        else
            m.pnls_w = m.pnls_w1
        end

        # Proposed QR decompositions in Step 1
        #       subject_qr_parallel: Function that performs QR decomposition for subject-level design matrices
        @inbounds for i = 1:m.n
            # Process QR decompostion of 'i'th subject on worker "wks_schedule[i]"
            futures[i] = remotecall(subject_qr_parallel_indep, wks_schedule[i],
                                    m.beta_w, m.alpha_w[:,i], m.gamma_w[i],
                                    m.Delta_U, m.Delta_V, m.y_mt_array[i], m.t)
            # A remote call returns a Future to its result immediately
            # Process that made call proceeds to its next operation while remote call happens somewhere else
        end
        @inbounds for i = 1:m.n
            wait(futures[i])
            # Computation on different workers do not finish at same time
            # Wait for all remote calls to finish by calling 'wait' on returned Futures
        end
        @inbounds for i = 1:m.n
            T = m.T
            p = m.p
            R_start = sum(m.R_array[1:(i-1)])
            R_end = sum(m.R_array[1:i])
            m.s_1_ir[(p*R_start+1):(p*R_end)],
            m.S_beta_1_ir[(p*R_start+1):(p*R_end),:],
            m.S_alpha_1_ir[(p*R_start+1):(p*R_end),:],
            m.R_1_ir[(p*R_start+1):(p*R_end),:],
            m.s_01_i[(p*(i-1)+1):(p*i),:],
            m.S_beta_01_i[(p*(i-1)+1):(p*i),:],
            m.R_2_i[(p*(i-1)+1):(p*i),:],
            m.s_0[(T*(i-1)+1):(T*i),:],
            m.S_beta_00[(T*(i-1)+1):(T*i),:] = fetch(futures[i])
            # Fetch results from Futures returned by remote call
        end

        # Calculate LLS estimators using matrices from QR decomposition
        qr_beta = qr(m.S_beta_00)
        m.beta_w1 .= vec(qr_beta.R \ ((qr_beta.Q' * m.s_0)[1:5,:]))
        @inbounds for i = 1:m.n
            p = m.p
            R_i = m.R_array[i]
            R_start = sum(m.R_array[1:(i-1)])
            m.alpha_w1[:,i] .= m.R_2_i[(p*(i-1)+1):(p*i),:] \ (m.s_01_i[(p*(i-1)+1):(p*i),:] - 
                                    m.S_beta_01_i[(p*(i-1)+1):(p*i),:] * m.beta_w1)
            @inbounds for r = 1:R_i
                m.gamma_w1[i][:,r] .= m.R_1_ir[(p*(R_start+r-1)+1):(p*(R_start+r)),:] \
                    (m.s_1_ir[(p*(R_start+r-1)+1):(p*(R_start+r))] - 
                    m.S_alpha_1_ir[(p*(R_start+r-1)+1):(p*(R_start+r)),:] * m.alpha_w1[:,i] -
                    m.S_beta_1_ir[(p*(R_start+r-1)+1):(p*(R_start+r)),:] * m.beta_w1)
            end
        end

        # Step halving procedure at end of each iteration of Gauss-Newton algorithm
        m.delta_beta .= (m.beta_w1 - m.beta_w)
        m.delta_alpha .= (m.alpha_w1 - m.alpha_w)
        for i=1:m.n
            m.delta_gamma[i] .= (m.gamma_w1[i] - m.gamma_w[i])
        end
        halv_iter = 1
        halv_err = 20
        while halv_iter <= halving_maxiter && halv_err >= 0
            m.beta_w1 .= m.beta_w + m.delta_beta
            m.alpha_w1 .= m.alpha_w + m.delta_alpha
            for i=1:m.n
                m.gamma_w1[i] .= m.gamma_w[i] + m.delta_gamma[i]
            end
            m.pnls_w1 = pnls_all(m.beta_w1, m.alpha_w1, m.gamma_w1)
            halv_err = m.pnls_w1 - m.pnls_w
            m.delta_beta .*= 0.5
            m.delta_alpha .*= 0.5
            for i=1:m.n
                m.delta_gamma[i] .*= 0.5
            end
            halv_iter += 1
        end

        # Calculate relative improvement in PNLS objective function
        m.gn_err = abs(m.pnls_w1 - m.pnls_w) / m.pnls_w
        m.gn_iter += 1
    end

    # End of Step 1: Overwrite fixed and random effects in NlmeModel object with estimations
    #                from current iteration
    m.beta .= m.beta_w1
    m.alpha .= m.alpha_w1
    for i=1:m.n
        m.gamma[i] .= m.gamma_w1[i]
    end

    # Taking absolute values of trial-specific fitted amplitude parameters as their
    # sign do not impact shape function (due to squaring) and it ensures a decrease in 
    # PNLS objective function
    m.beta .+= vec(mapslices(mean, m.alpha, dims = 2))
    m.alpha .-= vec(mapslices(mean, m.alpha, dims = 2))
    b_a_i = zeros(5,1)
    @inbounds for i = 1:m.n
        b_a_i .= m.beta + m.alpha[:,i]
            m.alpha[1,i] = abs(b_a_i[1]) - m.beta[1]
            m.alpha[3,i] = abs(b_a_i[3]) - m.beta[3]
        R_i = m.R_array[i]
        b_a_g_ir = zeros(5,1)
        @inbounds for r = 1:R_i
            b_a_g_ir .= b_a_i + m.gamma[i][:,r]
                m.gamma[i][1,r] = abs(b_a_g_ir[1]) - abs(b_a_i[1])
                m.gamma[i][3,r] = abs(b_a_g_ir[3]) - abs(b_a_i[3])
            
        end
    end

    # Step 2: Updating variance components by maximizing minorization function
    #       mnlme_mm_update_parallel: Function that calculates trial-level contributions to numerators
    #                                 and denominators for updating variance components via MM algorithm
    # Calculate subject-level components for MM update on different workers parallelly (Step 2b)
    @inbounds for i=1:m.n
        futures[i] = remotecall(mnlme_indep_mm_update_parallel, wks_schedule[i],
                                vec(m.beta), 
                                vec(m.alpha[:,i]), m.gamma[i],
                                m.y_mt_array[i], m.t,
                                m.sigma2, m.U_inv, m.U_half, m.V_inv)
    end
    # Wait for all workers to finish computation
    @inbounds for i=1:m.n
        wait(futures[i])
    end
    # Fetch results from Futures returned by remote call
    @inbounds for i=1:m.n
        m.mm_update_value[:,i] .= fetch(futures[i])
    end

    # Calculate numerators and denominators for updating variance components by summing up 
    # subject-specific contributions
    m.mm_update_sum .= sum(m.mm_update_value, dims = 2)
    m.logl = m.mm_update_sum[1]
    m.sigma_num = m.mm_update_sum[2]
    m.sigma_den = m.mm_update_sum[3]
    m.U_num .= m.mm_update_sum[4:8]
    m.U_den .= m.mm_update_sum[9:13]
    m.V_num .= m.mm_update_sum[14:18]
    m.V_den .= m.mm_update_sum[19:23]
    
    # Update/overwrite variance components in an element-wise fashion
    m.sigma2 = m.sigma2 * sqrt(m.sigma_num / m.sigma_den)
    m.U .= Diagonal(diag(m.U) .* sqrt.(m.U_num ./ m.U_den))
    m.V .= Diagonal(diag(m.V) .* sqrt.(m.V_num ./ m.V_den))

    # End of Step 2: Return approximated marginal log-likelihood evaluated at current
    #                model components estimation
    return(m.logl)

end



function subject_qr_parallel_indep(
    # current estimation of model components
    beta_w::Vector{FloatType},          # current estimation of fixed effects (vector of length p)
    alpha_w_i::Vector{FloatType},       # current estimation of i-th subject's subject-level random effects 
                                        # (vector of length p)
    gamma_w_i::Matrix{FloatType},       # current estimation of i-th subject's trial-level random effects
                                        # (matrix of dimension p*R_i)
    Delta_u::Diagonal{FloatType, Vector{FloatType}},
                                        # subject-level precision matrix derived from current estimation of variance components
                                        # (diagonal matrix of dimensioon p*p)
    Delta_v::Diagonal{FloatType, Vector{FloatType}},
                                        # trial-level precision matrix derived from current estimation of variance components
                                        # (diagonal matrix of dimensioon p*p)
    # data
    y_i::Matrix{FloatType},             # i-th subject's trial-level ERP response (matrix of dimension T*R_i)
    t::Vector{FloatType}                # trial time grid (vector of length T)
) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that performs QR decompositions for i-th subject-specific design 
    ##              matrices described in part A of Supplementary materials
    ## Definition:  T: number of trial time points
    ##              p: number of trial-level variance components (equals to number of shape 
    ##                 parameters in shape function)
    ##              R_i: number of trials of i-th subject
    ## Args:        see above
    ## Returns:     a tuple of coefficient matrices and constant vectors for LLS estimator derivation
    ##              s_1_ir, S_beta_1_ir, S_alpha_1_ir, R_1_ir, s_01_i, S_beta_01_i, R_2_i,
    ##              s_0_i, S_beta_00_i: notations given in part A of Supplementary materials
    #########################################################################################

    # Create empty variables with dimensions compatible to outcomes for better memory allocation
    R_i = size(gamma_w_i, 2)
    p = size(gamma_w_i, 1)
    T = size(y_i, 1)
    S_ir_list = Array{Matrix{FloatType}}(undef, R_i)
    s_ir_list = Array{Vector{FloatType}}(undef, R_i)
    R_1_ir_list = Array{Matrix{FloatType}}(undef, R_i)
    M_ir = Matrix{FloatType}(undef, T, p)
    # Current estimation of shape parameters for i-th subject
    phi_i = gamma_w_i .+ alpha_w_i .+ beta_w
    w_i = copy(y_i)

    # QR decomposition of trial-level design matrices
    @inbounds for r=1:R_i
        # Calculate first-order derivative of shape function over shape parameters evaluated at 
        # current estimation of shape parameters
        M_ir .= mu1(t, phi_i[:,r])      # mu1: first-order derivative of shape function with respective 
                                        #      to shape parameters (details see preparation_functions.jl)

        # Calculate adjusted trial-specific residual 
        w_i[:,r] .= y_i[:,r] .- mu(t, phi_i[:,r]) .+ M_ir * phi_i[:,r]

        # Design matrix of r-th trial-level random effects for i-th subject
        design_gamma_ir = Matrix([M_ir; Delta_v])   

        # QR decomposition of trial-level design matrix
        Q_ir, R_ir = qr(design_gamma_ir)
        R_1_ir_list[r] = R_ir

        # Transformed fixed and subject-level random effects design matrices and 
        # transformed trial-level residual (constant) vectors
        S_ir_list[r] = Q_ir' * [M_ir; Diagonal(zeros(5))]
        s_ir_list[r] = Q_ir' * [w_i[:,r]; zeros(5)]
    end

    # QR decomposition of subject-level design matrices 
    # Stack up matrices from trial-level decompositions to form subject-level design matrices
    S_i_design = vcat([matrix[(p+1):(T+p),:] for matrix in S_ir_list]...)
    s_i = vcat([vector[(p+1):(T+p)] for vector in s_ir_list]...)

    # QR decomposition of subject-level design matrix
    Q_i, R_i = qr([S_i_design; Delta_u])

    # Transformed fixed effects design matrix and transformed subject-level residual (constant) vector
    S_beta_i = Q_i' * [S_i_design; Diagonal(zeros(5))]
    s_i = Q_i' * [s_i; zeros(5)]
    return (s_1_ir_c = vcat([vector[1:p] for vector in s_ir_list]...),
            S_beta_1_ir_c = vcat([matrix[1:p,:] for matrix in S_ir_list]...),
            S_alpha_1_ir_c = vcat([matrix[1:p,:] for matrix in S_ir_list]...),
            R_1_ir_c = vcat([matrix for matrix in R_1_ir_list]...),
            s_01_i_c = s_i[1:p],
            S_beta_01_i_c = S_beta_i[1:p,:],
            R_2_i_c = R_i,
            s_0_i_c = s_i[(p+1):(T+p)],
            S_beta_00_i_c = S_beta_i[(p+1):(T+p), :])
end



function mnlme_indep_mm_update_parallel(
    beta::Vector{FloatType},                # current estimation of fixed effects (vector of length p)
    alpha_i::Vector{FloatType},             # current estimation of i-th subject's subject-level random effects (vector of length p)
    gamma_i::Matrix{FloatType},             # current estimation of i-th subject's trial-level random effects (matrix of dimension p*R_i)
    y_i::Matrix{FloatType},                 # trial-level ERP responses from i-th subject (matrix of dimension T*R_i)
    t::Vector{FloatType},                   # trial time grid (vector of length T)
    sigma2::FloatType,                      # current estimation of measurement error variance
    U_inv::Diagonal{FloatType, Vector{FloatType}},
                                            # inverse of current estimation of covariance matrix of subject-level random effects
                                            # (U^{-1} diagonal matrix of dimension p*p)
    U_half::Diagonal{FloatType, Vector{FloatType}},
                                            # 'square root' of current estimation of covariance matrix of subject-level random effects
                                            # (U^{1/2} diagonal matrix of dimension p*p)
    V_inv::Diagonal{FloatType, Vector{FloatType}}
                                            # inverse of current estimation of covariance matrix of trial-level random effects
                                            # (V^{-1} diagonal matrix of dimension p*p)
) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that calculates subject-specific contribution to numerators and 
    ##              denominators for updating variance components via MM algorithm for each 
    ##              single trial described in Section 3.2 and Supplementary A
    ## Definition:  T: number of trial time points
    ##              p: number of trial-level variance components (equals to number of shape 
    ##                 parameters in shape function)
    ##              R_i: number of trials of i-th subject
    ## Args:        see above
    ## Returns:     subject_specific contribution to numerators and denominators for updating 
    ##              variance components
    #########################################################################################

    # Create empty variables with dimensions compatible to outcomes for better memory allocation
    R_i = size(gamma_i, 2)
    T = size(t, 1)
    phi_i = gamma_i .+ alpha_i .+ beta
    e_i = copy(y_i)
    O_i_inv_list = Array{Matrix{FloatType}}(undef,R_i)
    M_i = Array{Matrix{FloatType}}(undef, R_i)
    log_det_Omega_i_inv = 0.0
    
    # Calculate trial-level components
    @inbounds for r=1:R_i
        # Trial-specific first-order derivative of shape function over shape parameters
        M_i[r] = mu1(t, phi_i[:,r])
        # Trial-specific adjusted residual
        e_i[:,r] .= y_i[:,r] .- mu(t, phi_i[:,r]) .+ M_i[r] * (gamma_i[:,r] + alpha_i)
        # Trial-specific contribution to approximated subject-level covariance matrices (derivation see Supplementary A)
        O_i_inv_list[r] = 1/sigma2 * Diagonal(ones(T)) - 
                        1/sigma2^2 * M_i[r] * inv(Matrix(V_inv + M_i[r]' * M_i[r]/sigma2)) * M_i[r]'
        # Trial-specific contribution to approxiamted marginal log-likelihood (derivation see Supplementary A)
        log_det_Omega_i_inv += log(det(O_i_inv_list[r]))
    end

    # Stack up trial-level components over trials to create subject-level components
    N_i_matrix = vcat(M_i...)
    M_i_matrix = blockdiag([sparse(matrix) for matrix in M_i]...)
    O_i_inv_matrix = blockdiag([sparse(matrix) for matrix in O_i_inv_list]...)

    # Calculate inverse of approxiamted subject-level covariance matrix via Woodbury matrix identity
    Omega_i_inv = O_i_inv_matrix - O_i_inv_matrix * N_i_matrix * 
                    inv(Matrix(U_inv + N_i_matrix' * O_i_inv_matrix * N_i_matrix)) * N_i_matrix' * O_i_inv_matrix
    log_det_Omega_i_inv +=  log(det(Diagonal(ones(5)) + U_half * N_i_matrix' * O_i_inv_matrix * N_i_matrix * U_half))
    e_i_vec = vec(e_i)
    Omega_e_i = Omega_i_inv * e_i_vec
    N_Omega_e_i = N_i_matrix' * Omega_e_i
    M_Omega_e_i = M_i_matrix' * Omega_e_i

    # Calculate subject-specific contribution to approximated marginal log-likelihood
    logl_marg = -0.5 * T * R_i * log(2*π*sigma2) - 0.5 * log_det_Omega_i_inv - 0.5 * e_i_vec' * Omega_e_i

    # Calculate subject-specific contribution to numerators and denominators for updating variance components
    sigma_num = sum(Omega_e_i.^2)
    sigma_den = sum(diag(Omega_i_inv))
    u_num = zeros(5)
    u_den = zeros(5)
    v_num = zeros(5)
    v_den = zeros(5)
    @inbounds for h=1:5
        u_num[h] = N_Omega_e_i[h]^2
        u_den[h] = N_i_matrix[:,h]' * Omega_i_inv * N_i_matrix[:,h]
        indices = collect(h:5:(5*(R_i-1)+h))
        v_num[h] = sum(M_Omega_e_i[indices].^2)
        v_den[h] = sum(diag(M_i_matrix[:,indices]' * Omega_i_inv * M_i_matrix[:,indices]))
    end
    return[logl_marg; sigma_num; sigma_den; u_num; u_den; v_num; v_den]
end




function start_mnlme_indep!(
    m::MnlmeModel{FloatType},       # MnlmeModel type object constructed by MnlmeModel_construct
                                    # list of values, including:
                                    #       y_mt_array: trial-level ERP responses for each subject
                                    #                   (array of matrices of dimension T*R_i, i=1,...,n)
                                    #       t: trial time grid (vector of length T)
                                    #       beta: current estimates of fixed effects (vector of length p)
                                    #       alpha: current estimation of subject-level random effects (matrix of p*n)
                                    #       gamma: current estiamtes of trial-level random effects for each subject 
                                    #              (array of matrices of dimension p*R_i, i=1,...,n)
                                    #       U: current estimates of subject-level variance components (diagonal matrix of dimension p*p)
                                    #       V: current estimates of trial-level variance components (diagonal matrix of dimension p*p)
                                    #       sigma2: current estimate of measurement error variance (scalar)
                                    #       logl: current estimate of approximated marginal log-likelihood (scalar)
                                    # * see full list of values in 'nlme_classfiles.jl"
    # parameters for trial-level ERP component detection
    peak_range::Vector{FloatType},  # range of interval to search for peak-shaped ERP component (vector of length 2)
    dip_range::Vector{FloatType},   # range of interval to search for dip-shaped ERP component (vector of length 2)
    # parameters for subject-specific single-level NLME fitting
    gn_maxiter::Int,                # maximum number of iterations for Gauss-Newton algorithm in single-level NLME
    gn_err_threshold::FloatType,    # relative convergence threshold for Gauss-Newton algorithm in single-level NLME
    halving_maxiter::Int,           # maximum times of step-halving in single-level NLME
    maxiter::Int,                   # maximum number of iterations for iterative MM algorithm for single-level NLME
    logl_err_threshold::FloatType   # relative convergence threshold for MM algorithm for single-level NLME
) where FloatType <: AbstractFloat

    # Create empty variables with dimension compatible to target starting values for better memory allocation
    alpha_single_fit = Matrix{FloatType}(undef, m.p, m.n)
    gamma_single_fit = Array{Matrix{FloatType}}(undef, m.n)
    R_list = m.R_array
    for i=1:m.n
        gamma_single_fit[i] = Matrix{FloatType}(undef, m.p, R_list[i])
    end
    V_single_fit = Matrix{FloatType}(undef, m.p, m.n)
    sigma2_single_fit = Matrix{FloatType}(undef, 1, m.n)

    # For each subject, fit a single-level NLME model
    for subject_index = 1:m.n
        # Construct `NlmeModel` type variable (see definition in "nlme_classfiles.jl")
        subject_test_data = NlmeModel_construct(m.y_mt_array[subject_index], m.t, 5)

        # Calculate starting values for single-level NLME (see details in "NLME_MM_single.jl)
        start_nlme!(subject_test_data, peak_range, dip_range)

        # Fit single-level NLME with MM algorithm (see details in "NLME_MM_single.jl")
        fit_nlme_mm!(subject_test_data, gn_maxiter, gn_err_threshold, halving_maxiter, maxiter, logl_err_threshold)

        # Save subject-specific single-level model results for multi-level NLME starting values calculation
        alpha_single_fit[:,subject_index] .= subject_test_data.beta
        gamma_single_fit[subject_index] .= subject_test_data.gamma
        V_single_fit[:,subject_index] .= diag(subject_test_data.V)
        sigma2_single_fit[:,subject_index] .= subject_test_data.sigma2
    end

    # Calculate starting values for fixed effects, subject- and trial-level random effects, subject-
    # and trial-level variance components and the variance of measurement error 
    beta_start = vec(mean(alpha_single_fit, dims = 2))
    alpha_start = alpha_single_fit .- beta_start
    gamma_start = gamma_single_fit
    U_start = Diagonal(vec(mapslices(var, alpha_start, dims = 2)))
    V_start = Diagonal(vec(mapslices(var, hcat(gamma_start...), dims = 2)))
    sigma2_start = mean(sigma2_single_fit)

    # Overwrites model components in "MnlmeModel" variable with calculated starting values
    m.beta .= beta_start
    m.alpha .= alpha_start
    @inbounds for i in 1:m.n
        m.gamma[i] .= gamma_start[i]
    end
    m.U .= U_start
    m.V .= V_start
    m.sigma2 = sigma2_start
end






function update_mnlme_unstr_mm!(
    m::MnlmeUnstrModel{FloatType},  # MnlmeUnstrModel type object constructed by MnlmeUnstrModel_construct
                                    # list of values, including:
                                    #       y_mt_array: trial-level ERP responses for each subject
                                    #                   (array of matrices of dimension T*R_i, i=1,...,n)
                                    #       t: trial time grid (vector of length T)
                                    #       beta: current estimates of fixed effects (vector of length p)
                                    #       alpha: current estimation of subject-level random effects (matrix of p*n)
                                    #       gamma: current estiamtes of trial-level random effects for each subject 
                                    #              (array of matrices of dimension p*R_i, i=1,...,n)
                                    #       U: current estimates of subject-level covariance matrix (matrix of dimension p*p)
                                    #       V: current estimates of trial-level covariance matrix (matrix of dimension p*p)
                                    #       sigma2: current estimate of measurement error variance (scalar)
                                    #       logl: current estimate of approximated marginal log-likelihood (scalar)
                                    # * see full list of values in 'nlme_classfiles.jl"
    gn_maxiter::Int,                # maximum number of iterations for Gauss-Newton algorithm
    gn_err_threshold::FloatType,    # relative convergence threshold for Gauss-Newton algorithm
    halving_maxiter::Int            # maximum times of step-halving
) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that performs MM iteration for multi-level NLME model with 
    ##              unstructured random effects described in 'Modeling intra-individual inter-
    ##              trial EEG response variability in Autism Spectrum Disorder' Section 3.2, 
    ##              Algorithm 2, overwrites model components in MnlmeUnstrModel type object input, 
    ##              and returns approxiamted marginal log-likelihood evaluated at current 
    ##              estimation of model components
    ## Definition:  T: number of trial time points
    ##              n: number of subjects
    ##              R_i: number of trials for subject i, i=1,...,n
    ##              p: number of shape parameters (equals to number of variance components on 
    ##                 each level)
    ## Args:        see above
    ## Overwrites:  values in NlmeModel type object: 
    ##                  beta: estimated fixed effects (vector of length p)
    ##                  alpha: estimated subject-level random effects (matrix of dimension p*n)
    ##                  gamma: estimated trial-level random effects for each subject
    ##                         (array of matrices of dimension p*R_i, i=1,...,n)
    ##                  U: estimated subject-level covariance matrix (matrix of dimension p*p)
    ##                  V: estimated trial-level covariance matrix (matrix of dimension p*p)
    ##                  sigma2: estiamted variance of measurement error (scalar)
    ## Returns:     logl: approximated marginal log-likelihood evaluated at current model 
    ##                    components estimation
    #########################################################################################

    # Preparation for parallel computation: schedule computational nodes for subject-level jobs
    # Create storage spaces for results from parallel computational nodes
    futures = Vector{Future}(undef, m.n)
    # Assigns each subset to a worker
    wks_schedule = Vector{Int}(undef, m.n)
    wk_list = workers()
    wk = 1
    wk_max = size(wk_list,1)
    @inbounds for i = 1:m.n
        wks_schedule[i] = wk_list[wk]
        wk == wk_max ? wk = 1 : wk += 1
    end
    flush(stdout)
    print("wks_schedule = ", wks_schedule, "\n")

    # Step 1: Update fixed and random effects by solving PNLS via Gauss-Newton Algorithm
    m.beta_w .= m.beta              # beta_w: estimation of fixed effects from previous iteration
    m.beta_w1 .= m.beta             # beta_w1: estimation of fixed effects from current iteration
    m.alpha_w .= m.alpha            # alpha_w: estimation of subject-level random effects from previous iteration
    m.alpha_w1 .= m.alpha           # alpha_w1: estimation of subject-level random effects from current iteration
    for i=1:m.n
        m.gamma_w[i] .= m.gamma[i]  # gamma_w: estimation of trial-level random effects from previous iteration
        m.gamma_w1[i] .= m.gamma[i] # gamma_w1: estimation of trial-level random effects from current iteration
    end

    m.gn_iter = 1                   # gn_iter: iteration index of Gauss-Newton algorithm
    m.gn_err = 10.0                 # gn_err: relative improvement in PNLS objective function from last iteration
    m.pnls_w  = 0.0                 # pnls_w: PNLS objective function taking values at previous model components estimation
    m.pnls_w1 = 0.0                 # pnls_w1: PNLS objective function taking values at current model components estimation
    U_eigen = eigen(m.U)
    V_eigen = eigen(m.V)
    m.U_half .= U_eigen.vectors * Diagonal(sqrt.(U_eigen.values)) * U_eigen.vectors'
    m.U_inv .= U_eigen.vectors * Diagonal(1 ./ U_eigen.values) * U_eigen.vectors'
    m.Delta_U .= sqrt(m.sigma2) * U_eigen.vectors * Diagonal(1 ./ sqrt.(U_eigen.values)) * U_eigen.vectors'
    m.V_half .= V_eigen.vectors * Diagonal(sqrt.(V_eigen.values)) * V_eigen.vectors'
    m.V_inv .= V_eigen.vectors * Diagonal(1 ./ V_eigen.values) * V_eigen.vectors'
    m.Delta_V .= sqrt(m.sigma2) * V_eigen.vectors * Diagonal(1 ./ sqrt.(V_eigen.values)) * V_eigen.vectors'

    # PNLS objective function to be minimized using Gauss-Newton algorithm
    function pnls_all(beta, alpha, gamma;
                        y_mt_array = m.y_mt_array,
                        n = m.n,
                        t = m.t,
                        R_array = m.R_array,
                        Delta_u = m.Delta_U,
                        Delta_v = m.Delta_V)
        pnls = 0.0
        @inbounds for i = 1:n
            R_i = R_array[i]
            @inbounds for r = 1:R_i
                pnls += sum((y_mt_array[i][:,r] - mu(t, beta .+ alpha[:,i] .+ gamma[i][:,r])).^2) +
                        sum((Delta_v * gamma[i][:,r]).^2)       # mu: shape function 
                                                                #     (details see preparation_functions.jl)
            end
            pnls += sum((Delta_u * alpha[:,i]).^2)
        end
        return pnls
    end

    # Step 1a: Perform iterative Gauss-Newton algorithm
    while m.gn_iter <= gn_maxiter && m.gn_err >= gn_err_threshold
        m.beta_w .= m.beta_w1
        m.alpha_w .= m.alpha_w1
        @inbounds for i=1:m.n
            m.gamma_w[i] .= m.gamma_w1[i]
        end

        if m.gn_iter == 1
            m.pnls_w = pnls_all(m.beta_w, m.alpha_w, m.gamma_w)
        else
            m.pnls_w = m.pnls_w1
        end

        # Proposed QR decompositions in Step 1
        #       subject_qr_parallel: Function that performs QR decomposition for subject-level design matrices
        @inbounds for i = 1:m.n
            # Process QR decompostion of 'i'th subject on worker "wks_schedule[i]"
            futures[i] = remotecall(subject_qr_parallel_unstr, wks_schedule[i],
                                    m.beta_w, m.alpha_w[:,i], m.gamma_w[i],
                                    m.Delta_U, m.Delta_V, m.y_mt_array[i], m.t)
            # A remote call returns a Future to its result immediately
            # Process that made call proceeds to its next operation while remote call happens somewhere else
        end
        @inbounds for i = 1:m.n
            wait(futures[i])
            # Computation on different workers do not finish at same time
            # Wait for all remote calls to finish by calling 'wait' on returned Futures
        end
        @inbounds for i = 1:m.n
            T = m.T
            p = m.p
            R_start = sum(m.R_array[1:(i-1)])
            R_end = sum(m.R_array[1:i])
            m.s_1_ir[(p*R_start+1):(p*R_end)],
            m.S_beta_1_ir[(p*R_start+1):(p*R_end),:],
            m.S_alpha_1_ir[(p*R_start+1):(p*R_end),:],
            m.R_1_ir[(p*R_start+1):(p*R_end),:],
            m.s_01_i[(p*(i-1)+1):(p*i),:],
            m.S_beta_01_i[(p*(i-1)+1):(p*i),:],
            m.R_2_i[(p*(i-1)+1):(p*i),:],
            m.s_0[(T*(i-1)+1):(T*i),:],
            m.S_beta_00[(T*(i-1)+1):(T*i),:] = fetch(futures[i])
            # Fetch results from Futures returned by remote call
        end

        # Calculate LLS estimators using matrices from QR decomposition
        qr_beta = qr(m.S_beta_00)
        m.beta_w1 .= vec(qr_beta.R \ ((qr_beta.Q' * m.s_0)[1:5,:]))
        @inbounds for i = 1:m.n
            p = m.p
            R_i = m.R_array[i]
            R_start = sum(m.R_array[1:(i-1)])
            m.alpha_w1[:,i] .= m.R_2_i[(p*(i-1)+1):(p*i),:] \ (m.s_01_i[(p*(i-1)+1):(p*i),:] - 
                                    m.S_beta_01_i[(p*(i-1)+1):(p*i),:] * m.beta_w1)
            @inbounds for r = 1:R_i
                m.gamma_w1[i][:,r] .= m.R_1_ir[(p*(R_start+r-1)+1):(p*(R_start+r)),:] \
                    (m.s_1_ir[(p*(R_start+r-1)+1):(p*(R_start+r))] - 
                    m.S_alpha_1_ir[(p*(R_start+r-1)+1):(p*(R_start+r)),:] * m.alpha_w1[:,i] -
                    m.S_beta_1_ir[(p*(R_start+r-1)+1):(p*(R_start+r)),:] * m.beta_w1)
            end
        end

        # Step halving procedure at end of each iteration of Gauss-Newton algorithm
        m.delta_beta .= (m.beta_w1 - m.beta_w)
        m.delta_alpha .= (m.alpha_w1 - m.alpha_w)
        for i=1:m.n
            m.delta_gamma[i] .= (m.gamma_w1[i] - m.gamma_w[i])
        end
        halv_iter = 1
        halv_err = 20
        while halv_iter <= halving_maxiter && halv_err >= 0
            m.beta_w1 .= m.beta_w + m.delta_beta
            m.alpha_w1 .= m.alpha_w + m.delta_alpha
            for i=1:m.n
                m.gamma_w1[i] .= m.gamma_w[i] + m.delta_gamma[i]
            end
            m.pnls_w1 = pnls_all(m.beta_w1, m.alpha_w1, m.gamma_w1)
            halv_err = m.pnls_w1 - m.pnls_w
            m.delta_beta .*= 0.5
            m.delta_alpha .*= 0.5
            for i=1:m.n
                m.delta_gamma[i] .*= 0.5
            end
            halv_iter += 1
        end

        # Calculate relative improvement in PNLS objective function
        m.gn_err = abs(m.pnls_w1 - m.pnls_w) / m.pnls_w
        m.gn_iter += 1
    end

    # End of Step 1: Overwrite fixed and random effects in NlmeModel object with estimations
    #                from current iteration
    m.beta .= m.beta_w1
    m.alpha .= m.alpha_w1
    for i=1:m.n
        m.gamma[i] .= m.gamma_w1[i]
    end

    # Taking absolute values of trial-specific fitted amplitude parameters as their
    # sign do not impact shape function (due to squaring) and it ensures a decrease in 
    # PNLS objective function
    m.beta .+= vec(mapslices(mean, m.alpha, dims = 2))
    m.alpha .-= vec(mapslices(mean, m.alpha, dims = 2))
    b_a_i = zeros(5,1)
    @inbounds for i = 1:m.n
        b_a_i .= m.beta + m.alpha[:,i]
            m.alpha[1,i] = abs(b_a_i[1]) - m.beta[1]
            m.alpha[3,i] = abs(b_a_i[3]) - m.beta[3]
        R_i = m.R_array[i]
        b_a_g_ir = zeros(5,1)
        @inbounds for r = 1:R_i
            b_a_g_ir .= b_a_i + m.gamma[i][:,r]
                m.gamma[i][1,r] = abs(b_a_g_ir[1]) - abs(b_a_i[1])
                m.gamma[i][3,r] = abs(b_a_g_ir[3]) - abs(b_a_i[3])
            
        end
    end

    # Step 2: Updating variance components by maximizing minorization function
    #       mnlme_unstr_mm_update_parallel: Function that calculates trial-level contributions to numerators
    #                                 and denominators for updating variance components via MM algorithm
    # Calculate subject-level components for MM update on different workers parallelly (Step 2b)
    @inbounds for i=1:m.n
        futures[i] = remotecall(mnlme_unstr_mm_update_parallel, wks_schedule[i],
                                vec(m.beta), 
                                vec(m.alpha[:,i]), m.gamma[i],
                                m.y_mt_array[i], m.t,
                                m.sigma2, m.U, m.U_inv, m.U_half, m.V, m.V_inv)
    end
    # Wait for all workers to finish computation
    @inbounds for i=1:m.n
        wait(futures[i])
    end
    # Fetch results from Futures returned by remote call
    @inbounds for i=1:m.n
        m.mm_update_value[:,i],
        m.rr_seris[(5*i-4):(5*i), :],
        m.n_omega_n_series[(5*i-4):(5*i), :],
        m.ss_series[(5*i-4):(5*i), :],
        m.m_omega_m_series[(5*i-4):(5*i), :] = fetch(futures[i])
    end

    # Calculate numerators and denominators for updating variance components by summing up 
    # subject-specific contributions
    m.mm_update_sum .= sum(m.mm_update_value, dims = 2)
    m.logl = m.mm_update_sum[1]
    m.rr_sum .= m.rr_seris[1:5, :]
    m.n_omega_n_sum .= m.n_omega_n_series[1:5,:]
    m.ss_sum .= m.ss_series[1:5,:]
    m.m_omega_m_sum .= m.m_omega_m_series[1:5,:]
    @inbounds for i=2:m.n
        m.rr_sum .+= m.rr_seris[(5*i-4):(5*i), :]
        m.n_omega_n_sum .+= m.n_omega_n_series[(5*i-4):(5*i), :]
        m.ss_sum .+= m.ss_series[(5*i-4):(5*i), :]
        m.m_omega_m_sum .+= m.m_omega_m_series[(5*i-4):(5*i), :]
    end
    m.rr_sum .= 0.5 * (m.rr_sum' + m.rr_sum)
    m.n_omega_n_sum .= 0.5 * (m.n_omega_n_sum' + m.n_omega_n_sum)
    m.ss_sum .= 0.5 * (m.ss_sum' + m.ss_sum)
    m.m_omega_m_sum .= 0.5 * (m.m_omega_m_sum' + m.m_omega_m_sum)

    # Update/overwrite variance of measurement error in an element-wise fashion
    m.sigma2 = m.sigma2 * sqrt(m.mm_update_sum[2] / m.mm_update_sum[3])

    # Update/overwrite covariance matrices via explicit formula
    L_u = cholesky(m.n_omega_n_sum).L
    L_u_inv = inv(L_u)
    L_rr_L = L_u' * m.rr_sum * L_u
    L_rr_L_eigen = eigen(L_rr_L)
    L_rr_L_half = L_rr_L_eigen.vectors * Diagonal(sqrt.(L_rr_L_eigen.values)) * L_rr_L_eigen.vectors'
    m.U .= L_u_inv' * L_rr_L_half * L_u_inv
    L_v = cholesky(m.m_omega_m_sum).L
    L_v_inv = inv(L_v)
    L_ss_L = L_v' * m.ss_sum * L_v
    L_ss_L_eigen = eigen(L_ss_L)
    L_ss_L_half = L_ss_L_eigen.vectors * Diagonal(sqrt.(L_ss_L_eigen.values)) * L_ss_L_eigen.vectors'
    m.V .= L_v_inv' * L_ss_L_half * L_v_inv

    # End of Step 2: Return approximated marginal log-likelihood evaluated at current
    #                model components estimation
    return(m.logl)

end





function subject_qr_parallel_unstr(
    # current estimation of model components
    beta_w::Vector{FloatType},          # current estimation of fixed effects (vector of length p)
    alpha_w_i::Vector{FloatType},       # current estimation of i-th subject's subject-level random effects 
                                        # (vector of length p)
    gamma_w_i::Matrix{FloatType},       # current estimation of i-th subject's trial-level random effects
                                        # (matrix of dimension p*R_i)
    Delta_u::Matrix{FloatType},         # subject-level precision matrix derived from current estimation of variance components
                                        # (matrix of dimensioon p*p)
    Delta_v::Matrix{FloatType},         # trial-level precision matrix derived from current estimation of variance components
                                        # (matrix of dimensioon p*p)
    # data
    y_i::Matrix{FloatType},             # i-th subject's trial-level ERP response (matrix of dimension T*R_i)
    t::Vector{FloatType}                # trial time grid (vector of length T)
) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that performs QR decompositions for i-th subject-specific design 
    ##              matrices described in part A of Supplementary materials
    ## Definition:  T: number of trial time points
    ##              p: number of trial-level variance components (equals to number of shape 
    ##                 parameters in shape function)
    ##              R_i: number of trials of i-th subject
    ## Args:        see above
    ## Returns:     a tuple of coefficient matrices and constant vectors for LLS estimator derivation
    ##              s_1_ir, S_beta_1_ir, S_alpha_1_ir, R_1_ir, s_01_i, S_beta_01_i, R_2_i,
    ##              s_0_i, S_beta_00_i: notations given in part A of Supplementary materials
    #########################################################################################

    # Create empty variables with dimensions compatible to outcomes for better memory allocation
    R_i = size(gamma_w_i, 2)
    p = size(gamma_w_i, 1)
    T = size(y_i, 1)
    S_ir_list = Array{Matrix{FloatType}}(undef, R_i)
    s_ir_list = Array{Vector{FloatType}}(undef, R_i)
    R_1_ir_list = Array{Matrix{FloatType}}(undef, R_i)
    M_ir = Matrix{FloatType}(undef, T, p)
    # Current estimation of shape parameters for i-th subject
    phi_i = gamma_w_i .+ alpha_w_i .+ beta_w
    w_i = copy(y_i)

    # QR decomposition of trial-level design matrices
    @inbounds for r=1:R_i
        # Calculate first-order derivative of shape function over shape parameters evaluated at 
        # current estimation of shape parameters
        M_ir .= mu1(t, phi_i[:,r])      # mu1: first-order derivative of shape function with respective 
                                        #      to shape parameters (details see preparation_functions.jl)

        # Calculate adjusted trial-specific residual 
        w_i[:,r] .= y_i[:,r] .- mu(t, phi_i[:,r]) .+ M_ir * phi_i[:,r]

        # Design matrix of r-th trial-level random effects for i-th subject
        design_gamma_ir = Matrix([M_ir; Delta_v])   

        # QR decomposition of trial-level design matrix
        Q_ir, R_ir = qr(design_gamma_ir)
        R_1_ir_list[r] = R_ir

        # Transformed fixed and subject-level random effects design matrices and 
        # transformed trial-level residual (constant) vectors
        S_ir_list[r] = Q_ir' * [M_ir; Diagonal(zeros(5))]
        s_ir_list[r] = Q_ir' * [w_i[:,r]; zeros(5)]
    end

    # QR decomposition of subject-level design matrices 
    # Stack up matrices from trial-level decompositions to form subject-level design matrices
    S_i_design = vcat([matrix[(p+1):(T+p),:] for matrix in S_ir_list]...)
    s_i = vcat([vector[(p+1):(T+p)] for vector in s_ir_list]...)

    # QR decomposition of subject-level design matrix
    Q_i, R_i = qr([S_i_design; Delta_u])

    # Transformed fixed effects design matrix and transformed subject-level residual (constant) vector
    S_beta_i = Q_i' * [S_i_design; Diagonal(zeros(5))]
    s_i = Q_i' * [s_i; zeros(5)]
    return (s_1_ir_c = vcat([vector[1:p] for vector in s_ir_list]...),
            S_beta_1_ir_c = vcat([matrix[1:p,:] for matrix in S_ir_list]...),
            S_alpha_1_ir_c = vcat([matrix[1:p,:] for matrix in S_ir_list]...),
            R_1_ir_c = vcat([matrix for matrix in R_1_ir_list]...),
            s_01_i_c = s_i[1:p],
            S_beta_01_i_c = S_beta_i[1:p,:],
            R_2_i_c = R_i,
            s_0_i_c = s_i[(p+1):(T+p)],
            S_beta_00_i_c = S_beta_i[(p+1):(T+p), :])
end







function mnlme_unstr_mm_update_parallel(
    beta::Vector{FloatType},                # current estimation of fixed effects (vector of length p)
    alpha_i::Vector{FloatType},             # current estimation of i-th subject's subject-level random effects (vector of length p)
    gamma_i::Matrix{FloatType},             # current estimation of i-th subject's trial-level random effects (matrix of dimension p*R_i)
    y_i::Matrix{FloatType},                 # trial-level ERP responses from i-th subject (matrix of dimension T*R_i)
    t::Vector{FloatType},                   # trial time grid (vector of length T)
    sigma2::FloatType,                      # current estimation of measurement error variance
    U::Matrix{FloatType},                   # current estimation of covariance matrix of subject-level random effects
    U_inv::Matrix{FloatType},               # inverse of current estimation of covariance matrix of subject-level random effects
                                            # (U^{-1} matrix of dimension p*p)
    U_half::Matrix{FloatType},              # 'square root' of current estimation of covariance matrix of subject-level random effects
                                            # (U^{1/2} matrix of dimension p*p)
    V::Matrix{FloatType},                   # current estimation of covariance matrix of trial-level random effects
    V_inv::Matrix{FloatType},               # inverse of current estimation of covariance matrix of trial-level random effects
                                            # (V^{-1} matrix of dimension p*p)
) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that calculates subject-specific contribution to numerators and 
    ##              denominators for updating variance components via MM algorithm for each 
    ##              single trial described in Section 3.2 and Supplementary A
    ## Definition:  T: number of trial time points
    ##              p: number of trial-level variance components (equals to number of shape 
    ##                 parameters in shape function)
    ##              R_i: number of trials of i-th subject
    ## Args:        see above
    ## Returns:     subject_specific contribution to numerators and denominators for updating 
    ##              variance components
    #########################################################################################

    # Create empty variables with dimensions compatible to outcomes for better memory allocation
    R_i = size(gamma_i, 2)
    T = size(t, 1)
    phi_i = gamma_i .+ alpha_i .+ beta
    e_i = copy(y_i)
    O_i_inv_list = Array{Matrix{FloatType}}(undef,R_i)
    M_i = Array{Matrix{FloatType}}(undef, R_i)
    log_det_Omega_i_inv = 0.0
    
    # Calculate trial-level components
    @inbounds for r=1:R_i
        # Trial-specific first-order derivative of shape function over shape parameters
        M_i[r] = mu1(t, phi_i[:,r])
        # Trial-specific adjusted residual
        e_i[:,r] .= y_i[:,r] .- mu(t, phi_i[:,r]) .+ M_i[r] * (gamma_i[:,r] + alpha_i)
        # Trial-specific contribution to approximated subject-level covariance matrices (derivation see Supplementary A)
        O_i_inv_list[r] = 1/sigma2 * Diagonal(ones(T)) - 
                        1/sigma2^2 * M_i[r] * inv(Matrix(V_inv + M_i[r]' * M_i[r]/sigma2)) * M_i[r]'
        # Trial-specific contribution to approxiamted marginal log-likelihood (derivation see Supplementary A)
        log_det_Omega_i_inv += log(det(O_i_inv_list[r]))
    end

    # Stack up trial-level components over trials to create subject-level components
    N_i_matrix = vcat(M_i...)
    O_i_inv_matrix = blockdiag([sparse(matrix) for matrix in O_i_inv_list]...)

    # Calculate inverse of approxiamted subject-level covariance matrix via Woodbury matrix identity
    Omega_i_inv = O_i_inv_matrix - O_i_inv_matrix * N_i_matrix * 
                    inv(Matrix(U_inv + N_i_matrix' * O_i_inv_matrix * N_i_matrix)) * N_i_matrix' * O_i_inv_matrix
    log_det_Omega_i_inv +=  log(det(Diagonal(ones(5)) + U_half * N_i_matrix' * O_i_inv_matrix * N_i_matrix * U_half))
    e_i_vec = vec(e_i)
    Omega_e_i = Omega_i_inv * e_i_vec

    # Calculate subject-specific contribution to approximated marginal log-likelihood
    logl_marg = -0.5 * T * R_i * log(2*π*sigma2) - 0.5 * log_det_Omega_i_inv - 0.5 * e_i_vec' * Omega_e_i

    # Calculate subject-specific contribution to numerators and denominators for updating variance components
    sigma_num = sum(Omega_e_i.^2)
    sigma_den = sum(diag(Omega_i_inv))
    rr_i = U * N_i_matrix' * Omega_e_i * Omega_e_i' * N_i_matrix * U
    n_omega_n_i = N_i_matrix' * Omega_i_inv * N_i_matrix
    ss_ir_sum = zeros(T, p, p)
    m_omega_m_ir_sum = zeros(T, p, p)
    @inbounds for r=1:R_i
        ss_ir_sum += V * M_i[r]' * Omega_e_i[(TT*r-TT+1):(TT*r),:] * Omega_e_i[(TT*r-TT+1):(TT*r),:]' * M_i[r] * V
        m_omega_m_ir_sum += M_i[r]' * Omega_i_inv[(TT*r-TT+1):TT*r,(TT*r-TT+1):TT*r] * M_i[r]
    end

    return (mm_update_value = [logl_marg; sigma_num; sigma_den],
            rr_i = rr_i,
            n_omega_n_i = n_omega_n_i,
            ss_ir_sum = ss_ir_sum,
            m_omega_m_ir_sum = m_omega_m_ir_sum)
end




function start_mnlme_unstr!(
    m::MnlmeUnstrModel{FloatType},  # MnlmeUnstrModel type object constructed by MnlmeUnstrModel_construct
                                    # list of values, including:
                                    #       y_mt_array: trial-level ERP responses for each subject
                                    #                   (array of matrices of dimension T*R_i, i=1,...,n)
                                    #       t: trial time grid (vector of length T)
                                    #       beta: current estimates of fixed effects (vector of length p)
                                    #       alpha: current estimation of subject-level random effects (matrix of p*n)
                                    #       gamma: current estiamtes of trial-level random effects for each subject 
                                    #              (array of matrices of dimension p*R_i, i=1,...,n)
                                    #       U: current estimates of subject-level variance components (diagonal matrix of dimension p*p)
                                    #       V: current estimates of trial-level variance components (diagonal matrix of dimension p*p)
                                    #       sigma2: current estimate of measurement error variance (scalar)
                                    #       logl: current estimate of approximated marginal log-likelihood (scalar)
                                    # * see full list of values in 'nlme_classfiles.jl"
    # parameters for trial-level ERP component detection
    peak_range::Vector{FloatType},  # range of interval to search for peak-shaped ERP component (vector of length 2)
    dip_range::Vector{FloatType},   # range of interval to search for dip-shaped ERP component (vector of length 2)
    # parameters for subject-specific single-level NLME fitting
    gn_maxiter::Int,                # maximum number of iterations for Gauss-Newton algorithm in single-level NLME
    gn_err_threshold::FloatType,    # relative convergence threshold for Gauss-Newton algorithm in single-level NLME
    halving_maxiter::Int,           # maximum times of step-halving in single-level NLME
    maxiter::Int,                   # maximum number of iterations for iterative MM algorithm for single-level NLME
    logl_err_threshold::FloatType   # relative convergence threshold for MM algorithm for single-level NLME
) where FloatType <: AbstractFloat

    # Create empty variables with dimension compatible to target starting values for better memory allocation
    alpha_single_fit = Matrix{FloatType}(undef, m.p, m.n)
    gamma_single_fit = Array{Matrix{FloatType}}(undef, m.n)
    R_list = m.R_array
    for i=1:m.n
        gamma_single_fit[i] = Matrix{FloatType}(undef, m.p, R_list[i])
    end
    sigma2_single_fit = Matrix{FloatType}(undef, 1, m.n)

    # For each subject, fit a single-level NLME model
    for subject_index = 1:m.n
        # Construct `NlmeModel` type variable (see definition in "nlme_classfiles.jl")
        subject_test_data = NlmeUnstrModel_construct(m.y_mt_array[subject_index], m.t, 5)

        # Calculate starting values for single-level NLME (see details in "NLME_MM_single.jl)
        start_nlme_unstr!(subject_test_data, peak_range, dip_range)

        # Fit single-level NLME with MM algorithm (see details in "NLME_MM_single.jl")
        fit_nlme_unstr_mm!(subject_test_data, gn_maxiter, gn_err_threshold, halving_maxiter, maxiter, logl_err_threshold)

        # Save subject-specific single-level model results for multi-level NLME starting values calculation
        alpha_single_fit[:,subject_index] .= subject_test_data.beta
        gamma_single_fit[subject_index] .= subject_test_data.gamma
        sigma2_single_fit[:,subject_index] .= subject_test_data.sigma2
    end

    # Calculate starting values for fixed effects, subject- and trial-level random effects, subject-
    # and trial-level variance components and the variance of measurement error 
    beta_start = vec(mean(alpha_single_fit, dims = 2))
    alpha_start = alpha_single_fit .- beta_start
    gamma_start = gamma_single_fit
    U_start = Matrix(Diagonal(vec(mapslices(var, alpha_start, dims = 2))))
    V_start = Matrix(Diagonal(vec(mapslices(var, hcat(gamma_start...), dims = 2))))
    sigma2_start = mean(sigma2_single_fit)

    # Overwrites model components in "MnlmeUnstrModel" variable with calculated starting values
    m.beta .= beta_start
    m.alpha .= alpha_start
    @inbounds for i in 1:m.n
        m.gamma[i] .= gamma_start[i]
    end
    m.U .= U_start
    m.V .= V_start
    m.sigma2 = sigma2_start
end
