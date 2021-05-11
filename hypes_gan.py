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
w_g = 0
beta1_g = .25
beta2_g = .9
c_g = 1

# optim params discriminator
learning_rate_d = .0011
w_d = 0
beta1_d = .25
beta2_d = .9
c_d = 1
