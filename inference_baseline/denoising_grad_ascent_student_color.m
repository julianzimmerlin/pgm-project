function [T, log_post] = denoising_grad_ascent_student_color(N, sigma, eta, iterations, init,alpha)
    N = double(N);
    T = double(init);
    log_post = zeros(3,iterations,1);
    for chan = 1:3
        N_chan = N(:,:,chan);
        T_chan = T(:,:,chan);
        for i = 1:iterations
            log_post(chan, i) = denoising_lp_student(T_chan,N_chan,sigma,alpha);
            grad_llh = denoising_grad_llh(T_chan,N_chan,sigma);
            grad_lp = mrf_grad_log_student_prior(T_chan,sigma,alpha);
            T_chan = T_chan + eta * (grad_llh + grad_lp);
        end
        log_post(chan, i) = denoising_lp_student(T_chan,N_chan,sigma,alpha);
        T(:,:,chan) = T_chan;
    end
end