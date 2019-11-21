function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
selection = zeros(num_labels,m);
for i = 1:m
    selection(y(i),i) = 1;
end

% for i = 1:m
%     J = J - log(sigmoid(sigmoid(X(i,:)*transpose(Theta1))*transpose(Theta2)))*selection(:,i)/m;
% end
% J = 

X = [ones(size(X,1),1),X];
out1 = sigmoid(X*transpose(Theta1));
out1 = [ones(size(out1,1),1),out1];

out2 = sigmoid(out1*transpose(Theta2));
out3 = log(out2);
out4 = log(1-out2);

J = -1/m*trace(out3*selection+out4*(1-selection));
J = J + trace(Theta1(:,2:end)*transpose(Theta1(:,2:end)))*lambda/(2*m);
J = J + trace(Theta2(:,2:end)*transpose(Theta2(:,2:end)))*lambda/(2*m);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

for i = 1:m
    Theta2_grad = Theta2_grad - transpose(transpose(selection(:,i)).*(1-out2(i,:)))*out1(i,:)/m;
    Theta2_grad = Theta2_grad + transpose(transpose(1-selection(:,i)).*out2(i,:)) * out1(i,:)/m;
end

Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);

for i = 1:m
    dout1 = zeros(size(out1(i,:)));
    dout1 = dout1-(transpose(selection(:,i)).*(1-out2(i,:)))*Theta2/m;
    dout1 = dout1+(transpose(1-selection(:,i)).*out2(i,:))*Theta2/m;
    Theta1_grad = Theta1_grad + transpose(dout1(2:end).*out1(i,2:end).*(1-out1(i,2:end)))*X(i,:);
end

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
