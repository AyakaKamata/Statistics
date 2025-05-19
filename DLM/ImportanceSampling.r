#CRAN: https://cran.r-project.org/web/packages/dlm/dlm.pdf
# install.packages("dlm")
library(dlm)
library(withr)
# Local Model, time-invariant
# Y = theta + v, v ~ N(0, V)
# theta = theta + w, w ~ N(0, W)
# theta_0 ~ N(m0, C0)
mod <- dlmModPoly(1,
                  dV = 2,
                  dW = 1,
                  m0 = 10,
                  C0 = 9)
mod2 <- dlmModPoly(1,
                   dV = 1,
                   dW = 10,
                   m0 = 10,
                   C0 = 9)
set.seed(23)
T <- 100
# a matrix of expected values of future states
# R list of variances of future states
# f matrix of expected values of future observations
# Q list of variances of future observations
# newStates(theta) list of matrices containing the simulated future values
# of the states. Each component of the list corresponds
# to one simulation.
# newObs same as newStates, but for the observations
simData <- with_preserve_seed(dlmForecast(
  mod = mod,
  nAhead = n,
  sampleNew = 1
))
simData2 <- with_preserve_seed(dlmForecast(
  mod = mod2,
  nAhead = n,
  sampleNew = 1
))
# print(simData)
y <- simData$newObs[[1]]
y2 <- simData2$newObs[[1]]
theta <- simData$newStates[[1]]
theta2 <- simData2$newStates[[1]]
# Plot the simulated data
library(ggplot2)
library(tidyr)
library(dplyr)
# Prepare data for ggplot
df <- data.frame(t = 1:length(y),
                 y = as.numeric(y),
                 state = as.numeric(theta)) %>%
  pivot_longer(
    cols = c("y", "state"),
    names_to = "variable",
    values_to = "value"
  )

df2 <- data.frame(t = 1:length(y2),
                  y = as.numeric(y2),
                  state = as.numeric(theta2)) %>%
  pivot_longer(
    cols = c("y", "state"),
    names_to = "variable",
    values_to = "value"
  )

library(gridExtra)
p1 <- ggplot(df, aes(x = t, y = value, color = variable)) +
  geom_line(size = 1) +
  labs(title = "Simulated Data and States, r=1/2") +
  theme_minimal() + theme(aspect.ratio = 1 / 2)
p2 <- ggplot(df2, aes(x = t, y = value, color = variable)) +
  geom_line(size = 1) +
  labs(title = "Simulated Data and States, r=10") +
  theme_minimal() + theme(aspect.ratio = 1 / 2)
grid.arrange(p1, p2, ncol = 1)

# SMC
N <- 1000
N_0 <- N / 2
pfOut <- matrix(NA_real_, T + 1, N)
wt <- matrix(NA_real_, T + 1, N)

# optimal importance sampling
# Var(theta_t|theta_(t-1),y_t)
importanceSd <- sqrt(drop(W(mod) - W(mod)^2 / (W(mod) + V(mod))))
# Var(y_t|theta_(t-1))
predSd <- sqrt(drop(W(mod) + V(mod)))

# init
pfOut[1, ] <- with_preserve_seed(rnorm(N, m0(mod), sqrt(C0(mod))))
wt[1, ] <- rep(1 / N, N)

# sampling
# generate particles at t-1
ls <- list()
for (t in 2:(T + 1)) {
  # generate particles
  means <- pfOut[t - 1, ] + as.numeric(W(mod)) * (y[t - 1] - pfOut[t - 1, ]) / (as.numeric(W(mod)) +
                                                                                  as.numeric(V(mod)))
  pfOut[t, ] <- with_preserve_seed(rnorm(N, means, importanceSd))
  
  # update the weights
  wt[t, ] <- dnorm(y[t - 1], pfOut[t - 1, ], predSd) * wt[t - 1, ]
  wt[t, ] <- wt[t, ] / sum(wt[t, ])
  
  # resample
  N.eff <- 1 / crossprod(wt[t, ])
  if (N.eff < N_0) {
    # resample
    ls[length(ls) + 1] <- t
    idx <- with_preserve_seed(sample(N, N, replace = TRUE, prob = wt[t, ]))
    pfOut[t, ] <- pfOut[t, idx]
    wt[t, ] <- 1 / N
  }
  
}

# Compare with Kalman filter
# y The input data, coerced to a matrix. This is present only if simplify is FALSE.
# mod The argument mod (possibly simplified).
# m Time series (or matrix) of filtered values of the state vectors. The series starts
# one time unit before the first observation.
# U.C See below.
# D.C Together with U.C, it gives the SVD of the variances of the estimation errors.
# The variance of m[t, ] − theta[t, ] is given by U.C[[t]] %*% diag(D.C[t,]^2)
# %*% t(U.C[[t]]).
modFilt <- dlmFilter(y, mod)
# theta_(1:T)
thetaHatKF <- modFilt$m[-1]
# calculate variance of error using SVD
sdKF <- with(modFilt, sqrt(unlist(dlmSvd2var(U.C, D.C))))[-1]

# calculate the weighted mean of particle theta
pfOut <- pfOut[-1, ]
wt <- wt[-1, ]
thetaHatPF <- sapply(1:T, function(i)
  weighted.mean(pfOut[i, ], wt[i, ]))
sdPF <- sapply(1:T, function(i)
  sqrt(weighted.mean((pfOut[i, ] - thetaHatPF[i])^2, wt[i, ])))

df_theta <- data.frame(
  t = 1:length(thetaHatKF),
  KF = as.numeric(thetaHatKF),
  PF = as.numeric(thetaHatPF),
  state = as.numeric(theta)
) %>%
  pivot_longer(
    cols = c("KF", "PF", "state"),
    names_to = "methods",
    values_to = "theta"
  )

df_sd <- data.frame(
  t = 1:length(sdKF),
  KF = as.numeric(sdKF),
  PF = as.numeric(sdPF),
  Var = rep(W(mod), length(sdKF))
) %>%
  pivot_longer(
    cols = c("KF", "PF", "Var"),
    names_to = "methods",
    values_to = "sd"
  )

wt_long <- wt %>%
  as.data.frame() %>%
  mutate(t = 1:nrow(.)) %>%
  pivot_longer(cols = -t,
               names_to = "particle",
               values_to = "wt")

p1 <- ggplot(df_theta, aes(x = t, y = theta, color = methods)) +
  geom_line(size = 0.5) +
  geom_vline(xintercept = unlist(ls),
             linetype = "dashed",
             color = "red") +
  labs(title = "Compare Theta") +
  theme_minimal() + theme(aspect.ratio = 1 / 2)
p2 <- ggplot(df_sd, aes(x = t, y = sd, color = methods)) +
  geom_line(size = 1) +
  geom_vline(xintercept = unlist(ls),
             linetype = "dashed",
             color = "red") +
  labs(title = "Compare Sd") +
  theme_minimal() + theme(aspect.ratio = 1 / 2)

# p3 <- ggplot(wt_long, aes(x = t, y = wt)) +
#   geom_point(size = 0.5) +
#   geom_vline(xintercept = unlist(ls),
#              linetype = "dashed",
#              color = "red") +
#   theme_minimal() + theme(aspect.ratio = 1 / 2)

grid.arrange(p1, p2, ncol = 1)
