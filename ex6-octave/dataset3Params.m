function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

% =========================================================================

##C = vals;
##sigma = vals;
##n = length(vals);
##error=Inf(n);
##
##for i=1:n
##  for j=1:n
##    model= svmTrain(X, y, C(i), @(x1,x2) gaussianKernel(x1,x2, sigma(j)));
##    pred = svmPredict(model,Xval);
##    error(i,j) = mean(double(pred ~= yval));
##  end
##end
##
##[~,p] = min(min(error,[],2));
##[~,q] = min(min(error,[],1));
##
##C=C(p);
##sigma=sigma(q);

##-----------------------------


vals = [0.01 0.03 0.1 0.3 1 3 10 30];
error_min = Inf;

for C_temp = vals
  for sigma_temp = vals
    model = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    if(error <= error_min)
      C = C_temp;
      sigma = sigma_temp;
      error_min = error;
    end
  end
end



end
