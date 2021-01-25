function out = student_t(x,y,sigma,alpha)
    out = (1 + 1/(2*sigma^2) * (x-y).^2).^(-alpha);
end