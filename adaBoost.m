function [model] = adaBoost(X,y,nBoosts,boostedClassifier)
% adaBoost(X,y,nBoosts,boostedClassifier)
%
% Description:
%       - Runs Boosting on a Model and Returns the New Model
%
% nboost:
%       - The number of boosting iterations 
% 
% boostedClassifier:
%       - Boosting is Preformed on This Classifier
%
% Author: Reza Asad (2014)

[N,d] = size(X);

model.nBoosts = nBoosts;
model.boostedClassifier = boostedClassifier;

% Initialize Weights
z = (1/N)*ones(N,1);

for k = 1:nBoosts
    % Train Weighted Classifier
    model.subModel{k} = boostedClassifier(X,y,z);
    
    % Compute Predictions
    yhat = model.subModel{k}.predict(model.subModel{k},X);
    
    % Compute Weighted Error Rate
    err = sum(z.*(y~=yhat));
    
    % Compute alpha
    alpha(k) = (1/2)*log((1-err)/err);
    
    % Update Weights
    z = z.*exp(-alpha(k)*y.*yhat);
    
    % Re-normalize Weights
    z = z/sum(z);
end

model.alpha = alpha;
model.predict = @predict;

end

function [y] = predict(model,X)
alpha = model.alpha;

for k = 1:length(model.subModel)
   y(:,k) = alpha(k)*model.subModel{k}.predict(model.subModel{k},X);
end
y = sign(sum(y,2));
end
