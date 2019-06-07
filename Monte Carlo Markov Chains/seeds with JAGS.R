# Fitting models with JAGS
# Dataset Seeds is available at http://www.openbugs.net/Examples/Seeds.html
# Classic-bugs tools https://sourceforge.net/projects/mcmc-jags/files/Examples/2.x/
# April 28, 2019
# Luis Da Silva.

library( "rjags" )
source("Rcheck.R")
library("tidyverse")

# Load data
data <- read.jagsdata("seeds-data.R")
df <- tibble(r = data$r, n= data$n, x1=data$x1, x2=data$x2)
df$p <- df$r / df$n
df

# Plot data
main_plot <- ggplot() + 
  geom_jitter(aes(x=x1, y=x2, color=p, size=p), data = df, width = 0.2, height = 0.2) +
  xlim(-0.2,1.2) + ylim(-0.2,1.2) + xlab("Type of seed") + ylab("Type of root extract")
main_plot

# Build standard logistic regression
logic1 <- glm(p ~ x1*x2, data=df, family="quasibinomial")
summary(logic1)

# Plot logistic regression
add.thresh.line <- function(betas, thrs=0.5) {
  x2 <- 0
  x1 <- seq(-0.2, 1.2, 0.01)
  thrs <- -log(1/thrs - 1)
  for (i in 1:length(x1)) {
    x2[i] <- (thrs-betas[1]-betas[2]*x1[i]) / (betas[4]+betas[3]*x1[i])
  }
  return (geom_line(aes(x=x1, y=x2)))
}

coef <- c(logic1$coefficients[1], logic1$coefficients[2], logic1$coefficients[4], logic1$coefficients[3])
main_plot + add.thresh.line(coef)

# Use JAGS to sample and fit the same model
# Adapted from Matin Plummer's Test1
jags_str <- "model {
   # Set prior distribution of coefficients
   beta0  ~ dnorm(0.0,1.0E-6);
   beta1  ~ dnorm(0.0,1.0E-6);
   beta2  ~ dnorm(0.0,1.0E-6);
   beta12 ~ dnorm(0.0,1.0E-6);
   
   # Error variance
   tau    ~ dgamma(1.0E-3,1.0E-3);    # 1/sigma^2
   sigma  <- 1.0/sqrt(tau);

   # Get N fits
   for (i in 1:N) {
      b[i]         ~ dnorm(0.0,tau);
      logit(p[i]) <- beta0 + beta1*x1[i] +beta2*x2[i] +
                     beta12*x1[i]*x2[i] + b[i];
      r[i]         ~ dbin(p[i],n[i]);
   }
}"

inits <- list("tau"=1, "beta0"=0, "beta1"=0, "beta2"=0, "beta12"=0)

jags.obj <- jags.model(textConnection(jags_str), data=data, 
                       inits = inits, n.chains = 10, n.adapt = 2500)

# Do a burn-in and sample
update(jags.obj, 2000)
samples <- coda.samples(jags.obj, c('beta0','beta1','beta2', 'beta12'), 200, 1)

# Autocorr plot
samples_matrix <- as.matrix(samples)
acf(samples_matrix[,1])

# I'm sampling again but with thining
samples <- coda.samples(jags.obj, c('beta0','beta1','beta2', 'beta12'), 200*30, 30)
samples_matrix <- as.matrix(samples)
acf(samples_matrix[,1])
summary(samples[1])
plot(samples[1])
gelman.plot(samples)

# Plot several betas
several_plot <- main_plot
idx <- sample( 1:nrow(samples_matrix), 5) 
for(i in idx) {
  several_plot <- several_plot + add.thresh.line(samples_matrix[i,] )
}
several_plot
