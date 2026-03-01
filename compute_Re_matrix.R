# This script computes a matrix showing how the decision threshold impact a given value of R.

library(tti)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
R <- as.numeric(args[1])
symptoms_to_appointement <- as.numeric(args[2])
appointment_to_test <- as.numeric(args[3])
test_to_result <- as.numeric(args[4])
t_incubation <- as.numeric(args[5])
sensitivity_file <- args[6]
output_file <- args[7]

# Common fixed parameters
alpha <- 0.19
kappa <- 0.32
eta <- 0.0
nu <- 1.0
q_days <- 7
offset <- -12.27
shape <- 21.13
rate <- 1.59
quarantine_days <- 7
isolation_days <- 10
rho_a <- 0
result_to_MCT = 2.264929 # mean time from case PCR result to manual contact notification, as determined using a fitted lognormal distribution (cascade_analysis.R)
DPT_succes_rate = 0.043
MCT_succes_rate = DPT_succes_rate / 60 * 616
omega <- MCT_succes_rate # Probability of being traced given (community or household) exposure
t_da <- 100 #t_ds + additional_delay_asymp
rho_test <- 0.9
rho_PCR <- 0.93

t_qs <- symptoms_to_appointement + appointment_to_test + test_to_result + result_to_MCT # Time delay from index cases symptom onset to quarantine of contacts.
t_qa <- 100 #t_qcs + additional_delay_asymp


compute_Re_ml <- function(sens_isolation, sens_release) {
  testing_rate <- sens_release-sens_isolation
  rho_ML <- sens_release

  t_ds <- symptoms_to_appointement + testing_rate * (appointment_to_test + test_to_result)

  rho <- rho_test * rho_PCR * rho_ML   # Isolation completeness: probability that a community infection is detected and effectively isolated by a test-trace-isolate program.

  df <- get_r_effective_df(
    alpha = alpha, # Probability of asymptomatic infection
    R = R,
    kappa = kappa, # Relative transmissibility of asymptomatic individual
    eta = eta, # All contacts are considered community contacts
    nu = nu, # No differentiation between household/community contacts
    t_ds = t_ds, # Time delay from symptom onset to isolation in detected symptomatic person
    t_da = t_da, # Time delay from symptom onset to isolation in detected asymptomatic person
    t_qcs = t_qs,
    t_qca = t_qa,
    t_qhs = t_qs,
    t_qha = t_qa,
    t_q = t_qs,
    omega_c = omega,
    omega_h = omega,
    omega_q = omega,
    quarantine_days = quarantine_days,
    isolation_days = isolation_days,
    rho_s = rho,
    rho_a = rho_a,
    t_incubation = t_incubation,
    offset = offset,
    shape = shape,
    rate = rate,
  )

  return(df$r_effective[1])  # Example: Replace with your actual formula
}


df <- read.csv(sensitivity_file)

unique_sens <- sort(unique(df$sensitivity))

Re_matrix <- matrix(NA, nrow = length(unique_sens), ncol = length(unique_sens),
                    dimnames = list(unique_sens, unique_sens))

for (i in seq_along(unique_sens)) {
  for (j in seq_along(unique_sens)) {
    if (unique_sens[i] < unique_sens[j]) {  # Ensure sens1 < sens2
      Re_matrix[i, j] <- compute_Re_ml(unique_sens[i], unique_sens[j])
    }
  }
}

write.csv(Re_matrix, output_file, row.names = TRUE)

