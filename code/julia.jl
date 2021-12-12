# Econometric Notes - Matlab code
#  Matteo Courthoud
#  01/01/2017

#import Pkg
#Pkg.add("LinearAlgebra")
#Pkg.add("Random")
#Pkg.add("Statistics")
#Pkg.add("Distributions")
#Pkg.add("DataFrames")
#Pkg.add("CSV")
#Pkg.add("Optim")

using LinearAlgebra
using Random
using Statistics
using Distributions
using DataFrames
using CSV
using Optim
using Plots


## Lecture 1

# Set seed
Random.seed!(123);

# Set the number of observations
n = 100;

# Set the dimension of X
k = 2;

# Draw a sample of explanatory variables
X = rand(Uniform(0,1), n, k);

# Draw the error term
σ = 1;
ε = rand(Normal(0,1), n, 1) * sqrt(σ);

# Set the parameters
β = [2; -1];

# Calculate the dependent variable
y = X*β + ε;

# Estimate beta
β_hat = inv(X'*X)*(X'*y)

# Equivalent but faster formulation
β_hat = (X'*X)\(X'*y);

# Even faster (but less intuitive) formulation
β_hat = X\y;

# Note that is generally not equivalent to Var(X)^-1 * Cov(X,y)...
β_wrong = inv(cov(X)) * cov(X, y)

# ...unless you include a constant
α = 3;
y1 = α .+ X*β + ε;
β_hat1 = [ones(n,1) X] \ y1
β_correct1 = inv(cov(X)) * cov(X, y1)

# Predicted y
y_hat = X*β_hat;

# Residuals
ε_hat = y - X*β_hat;

# Projection matrix
P = X * inv(X'*X) * X';

# Annihilator matrix
M = I - P;

# Leverage
h = diag(P);

# Biased variance estimator
σ_hat = ε_hat'*ε_hat / n;

# Unbiased estimator 1
σ_hat_2 = ε_hat'*ε_hat / (n-k);

# Unbiased estimator 2
σ_hat_3 = mean( ε_hat.^2 ./ (1 .- h) );

# R squared - uncentered
R2_uc = (y_hat'*y_hat)/ (y'*y);

# R squared
y_bar = mean(y);
R2 = ((y_hat .- y_bar)'*(y_hat .- y_bar))/ ((y .- y_bar)'*(y .- y_bar));

# Ideal variance of the OLS estimator
var_β = σ * inv(X'*X);

# Standard errors
std_β = sqrt.(diag(var_β));



## Lecture 2

# Set seed
Random.seed!(123);

# Homoskedastic standard errors
std_h = var(ε_hat) * inv(X'*X);

# HC0 variance and standard errors
Ω_hc0 = X' * (I(n) .* ε_hat.^2) * X;
std_hc0 = sqrt.(diag(inv(X'*X) * Ω_hc0 * inv(X'*X)))

# HC1 variance and standard errors
Ω_hc1 = n/(n-k) * X' * (I(n) .* ε_hat.^2) * X;
std_hc1 = sqrt.(diag(inv(X'*X) * Ω_hc1 * inv(X'*X)))

# HC2 variance and standard errors
Ω_hc2 = X' * (I(n) .* ε_hat.^2 ./ (1 .- h)) * X;
std_hc2 = sqrt.(diag(inv(X'*X) * Ω_hc2 * inv(X'*X)))

# HC3 variance and standard errors
Ω_hc3 = X' * (I(n) .* ε_hat.^2 ./ (1 .- h).^2) * X;
std_hc3 = sqrt.(diag(inv(X'*X) * Ω_hc3 * inv(X'*X)))

# Note what happens if you allow for full autocorrelation
omega_full = X'*ε_hat*ε_hat'*X;

# t-test for beta=0
t = abs.(β_hat ./ (std_hc1));

# p-value
p_val = 1 .- cdf.(Normal(0,1), t);

# F statistic of joint significance
SSR_u = ε_hat'*ε_hat;
SSR_r = y'*y;
F = (SSR_r - SSR_u)/k / (SSR_u/(n-k));

# 95# confidente intervals
conf_int = [β_hat - 1.96*std_hc1, β_hat + 1.96*std_hc1];



## Lecture 3

# Set seed
Random.seed!(123);

# Set the dimension of Z
l = 3;

# Draw instruments
Z = rand(Normal(0,1), n,l);

# Correlation matrix for error terms
S = [1 0.8; 0.8 1];

# Endogenous X
γ = [2 0; 0 -1; -1 3];
ε = rand(Normal(0,1), n, 2) * cholesky(S).U;
X = Z*γ .+ ε[:,1];

# Calculate y
y = X*β .+ ε[:,2];

# Estimate beta OLS
β_OLS = (X'*X)\(X'*y)

# IV: l=k=2 instruments
Z_IV = Z[:,1:k];
β_IV = (Z_IV'*X)\(Z_IV'*y)

# Calculate standard errors
ε_hat = y - X*β_IV;
V_NHC_IV = var(ε_hat) * inv(Z_IV'*X)*Z_IV'*Z_IV*inv(Z_IV'*X);
V_HC0_IV = inv(Z_IV'*X)*Z_IV' * (I(n) .* ε_hat.^2) * Z_IV*inv(Z_IV'*X);

# 2SLS: l=3 instruments
Pz = Z*inv(Z'*Z)*Z';
β_2SLS = (X'*Pz*X)\(X'*Pz*y)

# Calculate standard errors
ε_hat = y - X*β_2SLS;
V_NCH_2SLS = var(ε_hat) * inv(X'*Pz*X);
V_HC0_2SLS = inv(X'*Pz*X)*X'*Pz * (I(n) .* ε_hat.^2) *Pz*X*inv(X'*Pz*X);

# GMM 1-step: inefficient weighting matrix
W_1 = I(l);

# Objective function
gmm_1(b) = ( y - X*b )' * Z * W_1 *  Z' * ( y - X*b );

# Estimate GMM
β_gmm_1 = optimize(gmm_1, β_OLS).minimizer
ε_hat = y - X*β_gmm_1;

# Standard errors GMM
S_hat = Z' * (I(n) .* ε_hat.^2) * Z;
d_hat = -X'*Z;
V_gmm_1 = inv(d_hat * inv(S_hat) * d_hat');

# GMM 2-step: efficient weighting matrix
W_2 = inv(S_hat);

# Objective function
gmm_2(b) = ( y - X*b )' * Z * W_2 *  Z' * ( y - X*b );

# Estimate GMM
β_gmm_2 = optimize(gmm_2, β_OLS).minimizer

# Standard errors GMM
ε_hat = y - X*β_gmm_2;
S_hat = Z' * (I(n) .* ε_hat.^2) * Z;
d_hat = -X'*Z;
V_gmm_2 = inv(d_hat * inv(S_hat) * d_hat');



## Lecture 5

# Set seed
Random.seed!(123);

# Define Model Parameters
n = 1000;
α = 1;
β = .1;
γ = -4;

# Set number of simulations
n_simulations = 10000;

# Preallocate simulation results
α_short = zeros(n_simulations,1);
α_long = zeros(n_simulations,1);
α_pretest = zeros(n_simulations,1);

# Loop over simulations
for sim = 1:n_simulations

    # Generate Data
    Z = randn(n,1);
    X = γ*Z + randn(n,1);
    ε = randn(n,1);
    y = α*X + β*Z + ε;

    # Alpha estimate from long regression
    α_short[sim] = ((X'*X)\(X'*y))[1,1];

    # Alpha estimate from long regression
    estimates_temp = ([X Z]'*[X Z]) \ ([X Z]'*y);
    α_long[sim] = estimates_temp[1];

    # Compute test statistic
    e = y - [X Z]*estimates_temp;
    t = estimates_temp[2]/sqrt(var(ε)*([0 1]*inv([X Z]'*[X Z])*[0;1])[1]);

    # Pick estimate depending on test statistic
    if abs(t)>1.96
        α_pretest[sim] = α_long[sim];
    else
        α_pretest[sim] = α_short[sim];
    end
end

# Plot results
p1 = histogram(α_long, bins=30, title = "alpha long")
plot!([1], seriestype="vline", linewidth = 2)
p2 = histogram(α_short, bins=30, title = "alpha short")
plot!([1], seriestype="vline", linewidth = 2)
p3 = histogram(α_pretest, bins=30, title = "alpha pretest")
plot!([1], seriestype="vline", linewidth = 2)
p = plot(p1, p2, p3, layout=(1, 3), legend=false, size=(800, 300))
savefig(p, "../img/Fig_621.png")




# Sequence of sample sizes
n_sequence = [100,1000,10000,50000,100000];

# Initialize figure
figure
set(gcf,'position',[0,0,2000,1000])
n_fig = 1;

# Loop over data generating processes
for dgp=0:1
    for n=n_sequence
        for sim=1:n_simulations

            # Select beta according to dgp
            if dgp==0
                beta_n = beta;
            else
                beta_n = beta/sqrt(n)*sqrt(100);
            end

            # Generate Data
            Z = randn(n,1);
            X = γ*Z + randn(n,1);
            e = randn(n,1);
            y = alpha*X+ beta_n*Z + e;

            # Alpha estimate from long regression
            α_short[sim] = inv(X'*X)*X'*y;

            # Alpha estimate from long regression
            estimates_temp = inv([X,Z]'*[X,Z])*[X,Z]'*y;
            α_long[sim] = estimates_temp[1];

            # Compute test statistic
            e = y - [X,Z]*estimates_temp;
            t = estimates_temp[2]/sqrt(var(e)*([0,1]*inv([X,Z]'*[X,Z])*[0;1]));

            # Pick estimate depending on test statistic
            if abs(t)>1.96
                α_pretest[sim] = α_long[sim];
            else
                α_pretest[sim] = α_short[sim];
            end
        end

        # Add subplot
        subplot(2,length(n_sequence),n_fig)
        hist((α_pretest-alpha)*sqrt(n/100), 12)
        title(['beta=',num2str(beta_n)],'fontsize',10)
        xlabel(['n=',num2str(n)], 'fontsize',10)
        n_fig = n_fig+1;

    end
end
saveas(gcf,'figures/Fig_624.png')
