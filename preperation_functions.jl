###########################################################################################
## Description: Supporting functions used in the MM algorithms (Section 3) and simulated 
##              data generating (Section 4) for the single- and multi-level NLME described
##              in 'Modeling intra-individual inter-trial EEG response variability in 
##              Autism Spectrum Disorder'.
###########################################################################################
## Functions included:
##      1. find_erp: Function that detects the peak- or dip-shaped component from a single 
##                   curve based on the given searching interval and returns its latency and
##                   amplitude.
##      2. mu: The shape function.
##      3. mu1: The first-order derivative of the shape function with respective to the shape
##              parameters.
###########################################################################################



function find_erp(
    argval::Vector{FloatType},          # trial time grid (vector of length T)
    y::Vector{FloatType},               # a single curve (vector of length T)
    range::Vector{FloatType},           # range of searching interval of the target component (vector of length 2)
    limit::FloatType,                   # a constant determines how much the searching interval will be widened at the maximum
                                        # e.g. limit = 1 -> interval widened to (1+1=2) times the original at the maximum
                                        #      limit = 0.5 -> interval widened to (1+0.5=1.5) times the original at the maximum
    peak = true                         # shape of the target component (true = peak; false = dip)
    ) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that detects the peak- or dip-shaped component from a single 
    ##              curve based on the given searching interval.
    ## Definition:  T: number of trial time points
    ## args:        see above
    ## Returns:     arg: latency of the detected component
    ##              val: amplitude of the detected component
    #########################################################################################

    peak_ind = 1

    # peak = true: find the peak-shaped component = find the loval maximum in the searching interval
    if peak
        peak_interval = findall(x -> x >= range[1] && x <= range[2], argval)
        # Check if peak_interval is empty
        if isempty(peak_interval)
            # Handle the situation here, e.g., error message, set default values, etc.
            error("peak_interval is empty")
        end
        interval_length = length(peak_interval)
        move = 1
        # check if peak is on the boundary and iterate
        while true
            # identify maximum within interval
            peak_ind = argmax(y[peak_interval])
            # shift interval if maximum on boundary
            if peak_ind == interval_length
                if peak_interval[end] == length(argval) || move >= limit * interval_length
                    break
                else
                    peak_interval .+= 1
                    move += 1
                end
            elseif peak_ind == 1
                if peak_interval[1] == 1 || move >= limit * interval_length
                    break
                else
                    peak_interval .-= 1
                    move += 1
                end
            else
                break
            end
        end
        amp_ind = peak_interval[peak_ind]
        peak_time = argval[amp_ind]
        peak = y[amp_ind]

    # peak = false: find the dip-shaped component = find the local minimum in the searching interval
    else
        # find the dip
        peak_interval = findall(x -> x >= range[1] && x <= range[2], argval)
        interval_length = length(peak_interval)
        move = 1
        # check if peak is on the boundary and iterate
        while true
            # identify minimum within interval
            peak_ind = argmin(y[peak_interval])
            # shift interval if minimum on boundary
            if peak_ind == interval_length
                if peak_interval[end] == length(argval) || move >= limit * interval_length
                    break
                else
                    peak_interval .+= 1
                    move += 1
                end
            elseif peak_ind == 1
                if peak_interval[1] == 1 || move >= limit * interval_length
                    break
                else
                    peak_interval .-= 1
                    move += 1
                end
            else
                break
            end
        end
        amp_ind = peak_interval[peak_ind]
        peak_time = argval[amp_ind]
        peak = y[amp_ind]
    end
    feature = (arg = peak_time, val = peak)
    return feature
end




function mu(
    t::Vector{FloatType},           # time grid (vector of length T)
    phi::Vector{FloatType};         # shape parameters (vector of length p)
    width1 = 4.0,                   # width parameter of the dip-shaped (N75) component
    width2 = 4.0                    # width parameter of the peak-shaped (P100) component
                                    # width parameters can be changed to adapt different settings
) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Shape function that consists of two exponential kernels
    ## Definition:  T: number of time points in the time grid
    ##              p = 5: number of shape parameters
    ## args:        see above
    ## Returns:     y: a single curve showing the shape function evaluated at the given shape
    ##                 parameters and time grid (vector of length T)
    #########################################################################################

    T = size(t, 1)
    y = zeros(Float64, T)
    #  plug in the shape parameters into the shape function with two exponential kernels
    for i in 1:T
        y[i] += -phi[1]^2 * exp(-0.5 * (t[i]-phi[2])^2/width1^2) + 
                    phi[3]^2 * exp(-0.5 * (t[i]-phi[4])^2/width2^2) + phi[5]
    end
    return y
end




function mu1(
    t::Vector{FloatType},           # time grid (vector of length T)
    phi::Vector{FloatType};         # shape parameters (vector of length p)
    width1 = 4.0,                   # width parameter of the dip-shaped (N75) component
    width2 = 4.0                    # width parameter of the peak-shaped (P100) component
                                    # width parameters can be changed to adapt different settings
) where FloatType <: AbstractFloat

    #########################################################################################
    ## Description: Function that calculates the first-order derivative matrix of the shape
    ##              function with respective to the shape parameters
    ## Definition:  T: number of time points in the time grid
    ##              p: number of shape parameters
    ## args:        see above
    ## Returns:     y: the first-order derivative matrix of the shape function with respect to
    ##                 the shape parameters evaluated at the given shape parameters and time 
    ##                 grid (matrix of dimension T*p)
    #########################################################################################

    T = size(t, 1)
    y = zeros(Float64, T, 5)
    for i in 1:T
        exp1 = exp(-0.5 * (t[i] - phi[2])^2 / width1^2)
        exp2 = exp(-0.5 * (t[i] - phi[4])^2 / width2^2)
        y[i, 1] = -2 * phi[1] * exp1
        y[i, 2] = phi[1]^2 * exp1 * (phi[2] - t[i]) / width1^2
        y[i, 3] = 2 * phi[3] * exp2
        y[i, 4] = -phi[3]^2 * exp2 * (phi[4] - t[i]) / width2^2
        y[i, 5] = 1.0
    end
    return y

end
