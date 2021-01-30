%function O = demo_denoise_foe()
%DEMO_DENOISE_FOE   Image denoising demo with FoE model.
%
%   Author:  Stefan Roth, Department of Computer Science, Brown University
%   Contact: roth@cs.brown.edu
%   $Date: 2005-06-08 18:47:29 -0400 (Wed, 08 Jun 2005) $
%   $Revision: 70 $

% Copyright 2004,2005, Brown University, Providence, RI.
% 
%                         All Rights Reserved
% 
% Permission to use, copy, modify, and distribute this software and its
% documentation for any purpose other than its incorporation into a
% commercial product is hereby granted without fee, provided that the
% above copyright notice appear in all copies and that both that
% copyright notice and this permission notice appear in supporting
% documentation, and that the name of Brown University not be used in
% advertising or publicity pertaining to distribution of the software
% without specific, written prior permission.
% 
% BROWN UNIVERSITY DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
% INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
% PARTICULAR PURPOSE.  IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR
% ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

  % begin Julian Zimmerlin: Reshape our filters to match their format
  cliquesize=3;
  load('../filters.mat');
  load('../alphas.mat');
  alphas = alphas * 5e-7; % I rescale the alphas here instead of adjusting the learning rate
                          % so that I dont't have to retrain them 
  filters = V(:,1:end-1);
  mirrorfilters = filters(size(filters,1):-1:1, :);
  filters = reshape(filters, cliquesize, cliquesize, 3, []); 
  mirrorfilters = reshape(mirrorfilters, cliquesize, cliquesize, 3, []);
  % end Julian Zimmerlin
  
  % Peppers image
  I = double(imread('../images/castle.jpg'));

  % Add Gaussian noise
  sigma = 15;
  N = I + sigma * randn(size(I));
  
  % Perform 100 iterations of denoising
  O = denoise_foe(N, filters, mirrorfilters, alphas, sigma, 100, 62.5, I);
 
  %begin Julian Zimmerlin:  Show images
  %figure, imshow(uint8(I))
  %figure, imshow(uint8(N))
  %figure, imshow(uint8(O))
  %end Julian Zimmerlin
