function g = denoising_grad_llh(T,N,sigma)
    g = double(1/sigma^2 * (N-T));
end