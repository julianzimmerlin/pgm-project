function lp = denoising_lp_student(T, N, sigma, alpha)
    log_likelihood = -1/(2*sigma^2)*sum((T - N).^2, 'all');
    diff_hor = T(:,1:end-1) - T(:,2:end);
    diff_ver = T(1:end-1,:) - T(2:end,:);
    help_hor = 1 / (2*sigma^2) * diff_hor.^2;
    help_ver = 1 / (2*sigma^2) * diff_ver.^2;
    help2_hor = (1 + help_hor).^(-alpha);
    help2_ver = (1 + help_ver).^(-alpha);
    log_prior = sum(log(help2_hor), 'all') + sum(log(help2_ver),'all');
    lp = log_likelihood + log_prior;
end