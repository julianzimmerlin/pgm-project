function O = denoise_foe(N, filters, mirrorfilters, alphas, sigma, lambda, niters, delta_t, I)
%DENOISE_FOE   Image denoising with FoE model.
%   O = DENOISE_FOE(N, P, SIGMA, NITERS, DELTAT) performs image denoising
%   of image N using an FoE model P.  The algorithm assumes additive
%   Gaussian noise with standard deviation SIGMA.  The algorithm will
%   perform NITERS iterations with step size DELTAT.
%
%   O = DENOISE_FOE(N, P, SIGMA, NITERS, DELTAT, I) performs image denoising
%   as before.  Image I is the original image, which is used for
%   evaluation of the signal-to-noise ratio for the status messages.
%
%   Image N can either be a gray level image (0 .. 255), or an RGB
%   image (0 .. 255 in all channels).
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

  
  % Find appropriate lambda value for given sigma.  The sigma-lambda
  % pairs here are determined experimentally to give good denoising
  % performance with the 5x5 model.
  %sigmas  = [5 10 15 25];
  %lambdas = [0.4 0.5 0.5 0.5];
  %lambda  = interp1(sigmas, lambdas, sigma, 'spline');

  % Perform given number of denoising iterations.
  O = N;
  g = zeros(size(O));
  if size(sigma,2) == 3
      factors = zeros(size(O));
      factors(:,:,1) = sigma(1) * ones(size(O,1,2));
      factors(:,:,2) = sigma(2) * ones(size(O,1,2));
      factors(:,:,3) = sigma(3) * ones(size(O,1,2));
  end
  bestimg = O;
  bestpsnr = -100;
  for i = 1:niters
    
    % Print out status every iteration
    % PSNR output, if original image is given.
    if (nargin > 7)
      O_rgb = ycbcr2rgb(uint8(O));
      psn = psnr(O_rgb, I);
      fprintf('%d/%d iterations (PSNR=%2.2fdB)\n', i, niters, psn);
      if psn > bestpsnr
        bestpsnr = psn;
        bestimg = O;
      end
    else
      fprintf('%d/%d iterations\n', i, niters);
    end
    
    g = evaluate_foe_log_grad(filters, mirrorfilters, alphas, O);
    if size(sigma,2)==3
        O = O + delta_t * (g + (lambda ./ factors.^2) .* (N - O));
    else
        O = O + delta_t * (g + (lambda / sigma^2) * (N - O));
    end
  end
  O = bestimg;
end