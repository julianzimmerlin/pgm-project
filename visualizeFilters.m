% visualize filters
cliquesize=3;
load('filters_ycbcr.mat');
load('alphas.mat');
filters = V;
for i=1:size(filters,2)
    fil = filters(:,i);
    fil = reshape(fil, cliquesize, cliquesize, 3);
    fil=(fil-min(fil(:)))/(max(fil(:))-min(fil(:)));
    %subplot(size(filters,2),5 ,i), 
    figure, imshow(fil, 'InitialMagnification',5000)
end