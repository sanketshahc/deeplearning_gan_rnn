xout_size = 28 * 28 # Fixed
epochs = 50  # iterations
batch_size = 64 #

# z, hidden sizes for generator
z_size = 200
hs_g1 = 1024
hs_g2 = 2048
hs_g3 = 2048

# hidden sizes for discriminator
hs_d1= 1024
hs_d2= 1024
hs_d3= 512

# optim params generator
learning_rate_g = .0001
w_g = 0 # weight decay l2
beta1_g = .25 # adam optim
beta2_g = .9 # adam optim
c_g = 1 # clamp value for gradients
r = 5 # number repitions

# optim params discriminator
learning_rate_d = .0011
w_d = 0
beta1_d = .5
beta2_d = .9
c_d = 1 # clamp value for gradient norm
c_w = .1 # clamp value for weights (absolute)
# c_w = [0.1, 0.01, 0.001, 0.0001]
# Add the gradients from the all-real and all-fake batches
