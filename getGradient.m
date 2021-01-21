function grad_alphas = getGradient(filters,all_patches)
clique_size = 3;
patch_size = 7;
disp('filter size')
size(filters)
% Generate 50,000 random images of dimension =  patchsize x patchsize x 3
Y = randi([0,255],patch_size,patch_size,3,50000);
% initialize grad alphas to zero
grad_alphas = zeros(size(filters,2),1);

% iterate over all alphas
for k = 1:length(grad_alphas)
    % get gradient of sum of log(experts)
    grad_log_experts_alpha = 0;
    % iterate over all patches
    for i = 1:size(all_patches,4)
        % get all cliques from each patch and store them as column vectors
        % each column of cliques matrix is a clique
        cliques = getCliques(all_patches(:,:,:,i),[clique_size clique_size],[1 1]);
        sum_si_filter_clique = 0;
        
        for j = 1:size(cliques,2) % iterate over all the cliques
            sum_si_filter_clique = -log(1+0.5*dot(filters(:,k),cliques(:,j))^2) + sum_si_filter_clique;
        end
        grad_log_experts_alpha = grad_log_experts_alpha + sum_si_filter_clique;
    end
    % get gradient of log of partition function
    grad_log_partition_alpha = 0;
    % iterate over all random images
    for i = 1:size(Y,4)
    cliques = getCliques(Y(:,:,:,i),[clique_size clique_size],[1 1]);
    sum_si_filter_clique = 0;
        
        for j = 1:size(cliques,2) % iterate over all the cliques
            sum_si_filter_clique = -log(1+0.5*dot(filters(:,k),cliques(:,j))^2) + sum_si_filter_clique;
        end
        grad_log_partition_alpha = grad_log_partition_alpha + sum_si_filter_clique;
    end 
    grad_log_partition_alpha = grad_log_partition_alpha / size(Y,4);
    grad_alphas(k) = grad_log_experts_alpha - size(all_patches,4) * grad_log_partition_alpha;
    
end
    
    
end
    
        
