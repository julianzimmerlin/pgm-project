function g = mrf_grad_log_student_prior(T, sigma, alpha)
    shiftup = zeros(size(T));
    shiftdown = zeros(size(T));
    shiftleft = zeros(size(T));
    shiftright = zeros(size(T));
    shiftup(1:end-1,:) = T(2:end,:);
    shiftdown(2:end,:) = T(1:end-1,:);
    shiftleft(:,1:end-1) = T(:,2:end);
    shiftright(:,2:end) = T(:,1:end-1);
    
    student_up = student_t(T,shiftup,sigma,1);
    student_down = student_t(T,shiftdown,sigma,1);
    student_left = student_t(T,shiftleft,sigma,1);
    student_right = student_t(T,shiftright,sigma,1);
    
    diff_up = T - shiftup;
    diff_down = T - shiftdown;
    diff_left = T - shiftleft;
    diff_right = T - shiftright;
    diff_up(end, :) = zeros(1,size(diff_up,2));
    diff_down(1, :) = zeros(1,size(diff_down,2));
    diff_left(:, end) = zeros(size(diff_left,1),1);
    diff_right(:, 1) = zeros(size(diff_right,1),1);
    
    prod_up = student_up .* diff_up;
    prod_down = student_down .* diff_down;
    prod_left = student_left .* diff_left;
    prod_right = student_right .* diff_right;
    
    g = - alpha / sigma^2 * (prod_up+prod_down+prod_left+prod_right);
%     
%     student_plus1_up = student_t(T,shiftup,sigma,alpha+1);
%     student_plus1_down = student_t(T,shiftdown,sigma,alpha+1);
%     student_plus1_left = student_t(T,shiftleft,sigma,alpha+1);
%     student_plus1_right = student_t(T,shiftright,sigma,alpha+1);
%     
%     prod_up = (1 ./ student_up) .* student_plus1_up .* shiftup;
%     prod_down = -(1 ./ student_down) .* student_plus1_down .* shiftdown;
%     prod_left = (1 ./ student_left) .* student_plus1_left .* shiftleft;
%     prod_right = -(1 ./ student_right) .* student_plus1_right .* shiftright;
%     
%     g = - alpha / sigma^2 * (prod_up+prod_down+prod_left+prod_right);
end