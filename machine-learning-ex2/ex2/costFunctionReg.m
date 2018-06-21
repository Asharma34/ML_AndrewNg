function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
z = X * theta;
h = 1 ./ (1 + exp(-z));
first = -(y' * log(h));
second = -((1 - y)') * (log(1 - h));
theta_sqr = theta .^ 2;
J = (1 / m) * (first + second) + (lambda / (2 * m)) * (sum(theta_sqr) - theta_sqr(1));
error = h - y;
grad = (1 / m) * X' * error + (lambda / m) * theta;
grad(1) = grad(1) - (lambda / m) * theta(1);




% =============================================================

end
