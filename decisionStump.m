function [model] = decisionStump(X,y,z)
% decisionStump(X,y,z)
%
% Description:
%       - Runs a Decision Stump on a Data Set
%
% z:
%       - These Are the Weights Computed by the Boosting Function, i.e adaBoost 
% 
%
% Author: Reza Asad (2014)

N = size(X,1);
min_val = inf;
for k=[-1,1]
    for j =1:2
        for i=1:N
            split = X(i,j);
            yhat = k*sign(X(:,j)-split);
            error = sum(z.*(y~=yhat));
            if error < min_val
                min_val = error;
                best_variable = j;
                threshold = k;
                best_split = X(i,j);
            end
        end
        
    end
end

model.split = best_split;
model.split_variable = best_variable;
model.threshold = threshold;
model.predict = @predict;

end

function [yhat] = predict(model, Xhat)
yhat = model.threshold*sign(Xhat(:,model.split_variable)-model.split);
end





