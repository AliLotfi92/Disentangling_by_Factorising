# Disentangling by Factorising, Variational Auto-Encoders

PyTorch Implementation of Factorized VAE (Disentangled VAE)
Here is a implementation of Factorized VAE based on this paper (MNIST DataSet)
Link to the paper https://arxiv.org/pdf/1802.05983.pdf

So, with simple words, it is just VAE, however, with additional regularizer term, which is based on Vanilla GAN, that tries minizmize JS distance (not KL, the authors made a mistake) between $q(z)$ and $q(z_1)q(z_2)...q(z_d)$ 
So, in fact, we have the loss function (ELBO) as the same VAEs (kingma et.al 2014) PLUS KL divergence of the two terms mentioned above.
