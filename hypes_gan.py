xout_size = 28 * 28 # Fixed
epochs = 10  # iterations
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

# Add the gradients from the all-real and all-fake batches
# errD = -(errD_real - errD_fake)
#
# # Update D
# optimizerD.step()
#
# # Clipping weights
# for p in D.parameters():
#     p.data.clamp_(-0.01, 0.01)
#

# W objective for disc is to maximize difference between Dx - DGx (a higher score for real,
# lower score for fake)

# for gen it is just trying to maximize it's output. so for a desecent op, just negative 1 times
# these.....

# additionally, remove the sigmoid and just makie it normal ?

# clip weights of discriminator so l1 norm is not bigger than c...


#for 2.1:
# xout_size = 28 * 28 # Fixed
# epochs = 50  # iterations
# batch_size = 64 #
#
# # z, hidden sizes for generator
# z_size = 200
# hs_g1 = 1024
# hs_g2 = 2048
# hs_g3 = 2048
#
# # hidden sizes for discriminator
# hs_d1= 1024
# hs_d2= 1024
# hs_d3= 512
#
# # optim params generator
# learning_rate_g = .0001
# w_g = 0
# beta1_g = .25
# beta2_g = .9
# c_g = 1
#
# # optim params discriminator
# learning_rate_d = .0011
# w_d = .0001
# beta1_d = .5
# beta2_d = .9
# c_d = 2
