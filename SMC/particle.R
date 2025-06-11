# Guided Particle Filter
gpf <- function(N,
                Steps,
                Y,
                ESS_min,
                M0_sample,
                G0_func,
                Mt_sample,
                Gt_func,
                resampler) {
  # Storage for particles and weights
  X_particles <- matrix(NA, nrow = N, ncol = Steps)
  W_weights <- matrix(NA, nrow = N, ncol = Steps)
  w_hat <- matrix(NA, nrow = N, ncol = Steps)
  A_ancestors <- matrix(NA, nrow = N, ncol = Steps)
  is_resampled <- numeric(Steps)
  ESS <- numeric(Steps)
  
  # --- Initialization (t=0) ---
  # Sample initial particles
  X_particles[, 1] <- M0_sample(N, Y[1]) # Sample N particles from M0
  
  # Calculate initial weights
  w_hat[, 1] <-  sapply(X_particles[, 1], G0_func, y0 = Y[1])
  W_weights[, 1] <- w_hat[, 1] / sum(w_hat[, 1])
  
  A_ancestors[, 1] <- 1:N # Initial ancestors are self
  
  # --- Sequential Monte Carlo Loop (t=1 to T-1) ---
  for (t in 1:(Steps - 1)) {
    # Check ESS
    ESS[t] <- 1 / sum(W_weights[, t]^2)
    
    if (ESS [t]< ESS_min) {
      # Resampling step
      print(paste("Resampling at time step", t, "with ESS =", ESS))
      A_ancestors[, t] <- systematic_resampler(N, W_weights[, t])
      w_hat[, t] <- rep(1, N)
      is_resampled[t] <- 1
    } else {
      # No resampling, keep previous ancestors and weights
      A_ancestors[, t] <- 1:N
    }
    
    # Predict and update weights
    
    for (n in 1:N) {
      # Sample new particle from proposal M_t based on ancestor
      anc_idx <- A_ancestors[n, t]
      X_particles[n, t + 1] <- Mt_sample(X_particles[anc_idx, t], Y[t + 1])
      
      # Calculate Gt term (incorporating observation likelihood and state transition)
      gt_val <- Gt_func(prev_x = X_particles[anc_idx, t],
                        curr_x = X_particles[n, t + 1],
                        curr_y = Y[t + 1])
      
      # If gt_val is 0, stop execution
      if (gt_val == 0) {
        warning(
          paste(
            "gt_val became 0 at time step",
            t,
            "for particle",
            n,
            ". Stopping execution. Inappropriate proposal distribution may be the cause."
          )
        )
        return(list(
          particles = X_particles[, 1:(t + 1)],
          weights = W_weights[, 1:(t + 1)],
          ancestors = A_ancestors[, 1:(t + 1)]
        ))
      }
      
      # Update weight
      w_hat[n, t + 1] <- w_hat[n, t] * gt_val
    }
    
    # Normalize weights
    W_weights[, t + 1] <- w_hat[, t + 1] / sum(w_hat[, t + 1])
  }
  
  return(
    list(
      particles = X_particles,
      weights = W_weights,
      w_hat = w_hat,
      ancestors = A_ancestors,
      is_resampled = is_resampled,
      ESS=ESS
      
    )
  )
}

# --- Systematic Resampling Function ---
# Simulation parameters

systematic_resampler <- function(N, weights) {
  # Allocate ancestor indices
  ancestors <- numeric(N)
  # Compute cumulative weights
  v <- cumsum(weights * N)
  print(v)
  # Sample U from [0, 1)
  s <- runif(1, min = 0, max = 1)
  m <- 1
  for (n in 1:N) {
    # While current cumulative weight is less than s, increment m
    while (v[m] < s) {
      m <- m + 1
    }
    ancestors[n] <- m
    s <- s + 1
  }
  print(ancestors)
  
  return(ancestors)
}

#Test
# Test the GPF implementation with a simple linear Gaussian model(AR(1))

# Define the model parameters
PX0 <- function(sd_x, rho) {
  var_x <- sd_x^2
  # return(list(mean = 0, sd = sqrt(var_x / (1 - rho^2))))
  return(list(mean = 0, sd = sd_x))
}
PX <- function(prev_x, rho, sd_x) {
  return(list(mean = rho * prev_x, sd = sd_x))
}
PY <- function(curr_x, sd_y) {
  return(list(mean = curr_x, sd = sd_y))
}
M0 <- function(y0, sd_x, sd_y) {
  var_x <- sd_x^2
  var_y <- sd_y^2
  var = 1 / (1 / var_x + 1 / var_y)
  l = list(mean = var * (y0 / var_y), sd = sqrt(var))
  return(l)
}
M <- function(prev_x, curr_y, rho, sd_x, sd_y) {
  var_x <- sd_x^2
  var_y <- sd_y^2
  var = 1 / (1 / var_x + 1 / var_y)
  l = list(mean = ((rho * prev_x) / var_x + curr_y / var_y) * var,
           sd = sqrt(var))
  return(l)
}

# Example M0_sample
M0_sample <- function(N, y0) {
  m0 <- M0(y0, sd_x, sd_y)
  return(rnorm(N, m0$mean, m0$sd))
}

# Example Mt_sample
Mt_sample <- function(prev_x, curr_y) {
  m <- M(prev_x, curr_y, rho, sd_x, sd_y)
  return(rnorm(1, m$mean, m$sd))
}

# Example G0_func
G0_func <- function(x0, y0) {
  py <- PY(x0, sd_y)
  px <- PX0(sd_x, rho)
  m <- M0(y0, sd_x, sd_y)
  # Calculate the proposal density
  f <- dnorm(y0, py$mean, py$sd)
  P <- dnorm(x0, px$mean, px$sd)
  Proposal <- dnorm(x0, m$mean, m$sd)
  # Avoid division by zero
  if (Proposal == 0)
    return(0)
  return((f * P) / Proposal)
}

# Example Gt_func
Gt_func <- function(prev_x, curr_x, curr_y) {
  py <- PY(curr_x, sd_y)
  px <- PX(prev_x, rho, sd_x)
  m <- M(prev_x, curr_y, rho, sd_x, sd_y)
  # Calculate the proposal density
  f <- dnorm(curr_y, py$mean, py$sd)
  P <- dnorm(curr_x, px$mean, px$sd)
  Proposal <- dnorm(curr_x, m$mean, m$sd)
  # Avoid division by zero
  if (Proposal == 0)
    return(0)
  
  return((f * P) / Proposal)
}


# Function to generate test data
test_generator <- function(Steps, rho, sd_x, sd_y) {
  X <- numeric(Steps + 1)
  Y <- numeric(Steps)
  
  X[1] <- rnorm(1, PX0(sd_x, rho)$mean, PX0(sd_x, rho)$sd)
  for (t in 1:Steps) {
    py <- PY(X[t], sd_y)
    Y[t] <- rnorm(1, py$mean, py$sd)
    px <- PX(X[t], rho, sd_x)
    X[t + 1] <- rnorm(1, px$mean, px$sd)
  }
  return(Y)
}



observations <- test_generator(Steps = 100, rho=0.9 , sd_x=1 , sd_y=0.2)
N_particles <- 1000
T_steps <- 100
rho = 0.9
sd_x = 1
sd_y = 0.2
ESS_threshold <- N_particles / 2


# Run the GPF
set.seed(123) # for reproducibility
gpf_results <- gpf(
  N = N_particles,
  Steps = T_steps,
  Y = observations,
  ESS_min = ESS_threshold,
  M0_sample = M0_sample,
  G0_func = G0_func,
  Mt_sample = Mt_sample,
  Gt_func = Gt_func,
  resampler = systematic_resampler
)

estimate <- function(result, ypred_cal, xpred_cal) {
  E_filter <- colSums(result$weights * result$particles)
  d_ypred <- ypred_cal(result$w_hat, result$is_resampled)
  d_ymarginal <- cumprod(d_ypred)
  E_xpred <- xpred_cal(result$particles, result$w_hat,result$ancestors)
  
  return(
    list(
      E_filter = E_filter,
      d_ypred = d_ypred,
      d_ymarginal = d_ymarginal,
      E_xpred = E_xpred
    )
  )
}


ypred_cal <- function(w_hat, is_resampled) {
  Steps <- ncol(w_hat)
  N <- nrow(w_hat)
  ell_N_t <- numeric(Steps)
  
  ell_N_t[1] <- sum(gpf_results$w_hat[, 1]) / N
  
  
  for (t in 2:Steps) {
    if (is_resampled[t - 1] == 1) {
      ell_N_t[t] <- sum(w_hat[, t]) / sum(w_hat[, t - 1])
    }
  }
  return(ell_N_t)
}
xpred_cal <- function(particles, w_hat, ancestors) {
  Steps <- ncol(particles)
  N <- nrow(particles)
  predicted_expectations <- numeric(Steps)
  
  predicted_expectations[1] <- NA
  
  
  for (t in 2:Steps) {
    sum_weighted_psi_phi <- 0
    sum_w_hat_prev <- sum(w_hat[, t - 1])
    
    for (n in 1:N) {
      anc_idx <- ancestors[n, t - 1]
      px <- PX(particles[anc_idx, t - 1], rho, sd_x)
      P <- dnorm (particles[n, t], px$mean, px$sd)
      proposal <- M(particles[anc_idx, t - 1], particles[n, t], rho, sd_x, sd_y)
      Prop <- dnorm(particles[n, t], proposal$mean, proposal$sd)
      psi_val <- P / Prop
      
      sum_weighted_psi_phi <- sum_weighted_psi_phi +
        (w_hat[n, t - 1] * psi_val * particles[n, t])
    }
    predicted_expectations[t] <- (1 / sum_w_hat_prev) * sum_weighted_psi_phi
  }
  return(predicted_expectations)
}

estimated <- estimate(gpf_results, ypred_cal, xpred_cal)

library(ggplot2)
library(tidyr)


plot_data <- data.frame(
  Time = 1:T_steps,
  Observations = observations,
  Filtered_X = estimated$E_filter # estimated$E_filter がフィルタリングされたXの推定値であると仮定
)

ggplot(plot_data, aes(x = Time)) + # x軸をTimeに固定
  geom_line(aes(
    y = Observations,
    color = "Observations",
    linetype = "Observations"
  )) +
  geom_line(
    aes(
      y = Filtered_X,
      color = "Filtered X (E[X_t|Y_0:t])",
      linetype = "Filtered X (E[X_t|Y_0:t])"
    )
  ) +
  scale_color_manual(values = c(
    "Observations" = "blue",
    "Filtered X (E[X_t|Y_0:t])" = "red"
  )) +
  scale_linetype_manual(values = c(
    "Observations" = "solid",
    "Filtered X (E[X_t|Y_0:t])" = "dashed"
  )) +
  labs(
    title = "Observations vs. Filtered State Estimates",
    x = "Time Step",
    y = "Value",
    color = "Series",
    linetype = "Series"
  ) +
  theme_minimal() +
  theme(legend.position = "topright") 