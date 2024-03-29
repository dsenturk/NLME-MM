# NLME-MM

## Contents of this folder

- nlme_mm_tutorial.jl: A step-by-step implementation of MM algorithm for nonlinear mixed-effects models (NLME), and the associated procedures including simulated data generation and model fitting described in "Modeling intra-individual inter-trial EEG response variability in Autism Spectrum Disorder".

- nlme_mm.jl: Function of MM algorithm for single-level NLME given in Section 3.1 of "Modeling intra-individual inter-trial EEG response variability in Autism Spectrum Disorder", with parallel computation incorporated.

- mnlme_mm.jl: Function of MM algorithm for multi-level NLME given in Section 3.2 of "Modeling intra-individual inter-trial EEG response variability in Autism Spectrum Disorder", with parallel computation incorporated.

- simulation_generate.jl: Functions simulating single- and multi-level data set under the simulation design stated in Section 4 of "Modeling intra-individual inter-trial EEG response variability in Autism Spectrum Disorder".

- preperation_functions.jl: Other related functions including the nonlinear shape function and the function for the ERP component searching algorithm.

- nlme_classfiles.jl: Type definition and construction functions for `NlmeModel`, `NlmeUnstrModel`, `MnlmeModel` and `MnlmeUnstrModel` used in fitting single- and multi-level NLME models via MM algorithm.

## Introduction

The contents of this folder allow for the implementation of the MM algorithm for fitting single- and multi-level NLME proposed in "Modeling intra-individual inter-trial EEG response variability in Autism Spectrum Disorder". Users can simulate a sample data set (simulation_generate.jl) and apply the proposed MM algorithm to fit the NLME model (nlme_mm.jl, mnlme_mm.jl). Detailed instructions on how to perform the aforementioned procedures are included in NLME_MM_tutorial.jl.


## Requirements

The included Julia programs require Julia 1.9 and the packages listed in nlme_mm_tutorial.jl.

## Installation

Load the Julia program files into the global environment and install the required packages using commands in nlme_mm_tutorial.jl.
