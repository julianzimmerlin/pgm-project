function g = evaluate_log_grad(filters, mirrorfilters, alphas, x)
%EVALUATE_FOE_LOG_GRAD   Gradient of the log of an FoE distribution
%   EVALUATE_FOE_LOG_GRAD(P, X) computes the gradient with respect to the
%   image of the log density of an FoE distribution P at image X.  
%
%   Author:  Stefan Roth, Department of Computer Science, Brown University
%   Contact: roth@cs.brown.edu
%   $Date: 2005-06-09 12:08:55 -0400 (Thu, 09 Jun 2005) $
%   $Revision: 72 $

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

  
  nfilters = size(filters, 4);
  dim = size(filters, 1);
  dims = [dim,dim,3];

  % Transform filters if necessary
  Jp = filters;
  Jpm = mirrorfilters;

  g = zeros(size(x));
  
  for j = 1:nfilters
    % Filter mask, and mirrored filter mask.
    f = Jp(:,:,:,j);
    fm = Jpm(:,:,:,j);
    
    % Convolve and pad image appropriately
    tmp = zeros(size(x));
    tmp = convn(x, f, 'same');
      
    % Compute expert gradient
    tmp2 = convn(alphas(j) * tmp ./ (1 + 0.5 * tmp.^2), fm, 'same');
    g = g - tmp2;
  end
