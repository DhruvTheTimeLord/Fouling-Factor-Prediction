function [trainedModel, newrmse, yhat1, y1, percentoutlier,percentoutlierinitial,rsq] = baggedtree(expdata)
%%
inputTable = expdata;
predictorNames = {'DensityKgm3', 'Timehr', 'SurfaceTemperatureoC', 'FluidTemperatureoC', 'FluidVelocityms', 'EquivalentDiameterm', 'DissolvedOxygenppmw'};
predictors = inputTable(:, predictorNames);
response = inputTable.FoulingFactorm2KkW;
isCategoricalPredictor = [false, false, false, false, false, false, false];

% Train a regression model
% This code specifies all the model options and trains the model.
template = templateTree(...
    'MinLeafSize', 1, ...
    'NumVariablesToSample','all');
regressionEnsemble = fitrensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 11, ...
    'Learners', template);

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedModel.RequiredVariables = {'DensityKgm3', 'DissolvedOxygenppmw', 'EquivalentDiameterm', 'FluidTemperatureoC', 'FluidVelocityms', 'SurfaceTemperatureoC', 'Timehr'};
trainedModel.RegressionEnsemble = regressionEnsemble;
trainedModel.About = 'This struct is a trained model exported from Regression Learner R2021a.';
trainedModel.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = expdata;
predictorNames = {'DensityKgm3', 'Timehr', 'SurfaceTemperatureoC', 'FluidTemperatureoC', 'FluidVelocityms', 'EquivalentDiameterm', 'DissolvedOxygenppmw'};
predictors = inputTable(:, predictorNames);
response = inputTable.FoulingFactorm2KkW;
isCategoricalPredictor = [false, false, false, false, false, false, false];

% Set up holdout validation
cvp = cvpartition(size(response, 1), 'Holdout', 0.3);
trainingPredictors = predictors(cvp.training, :);
trainingResponse = response(cvp.training, :);
trainingIsCategoricalPredictor = isCategoricalPredictor;

% Train a regression model
% This code specifies all the model options and trains the model.
template = templateTree(...
    'MinLeafSize', 1, ...
    'NumVariablesToSample', 'all');
regressionEnsemble = fitrensemble(...
    trainingPredictors, ...
    trainingResponse, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 11, ...
    'Learners', template);

% Create the result struct with predict function
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
validationPredictFcn = @(x) ensemblePredictFcn(x);
trainingPredictFcn = @(x) ensemblePredictFcn(x);

% Compute validation predictions
validationPredictors = predictors(cvp.test, :);
validationResponse = response(cvp.test, :);
validationPredictions = validationPredictFcn(validationPredictors);
trainingPredictions = trainingPredictFcn(trainingPredictors);
% Compute validation RMSE
isNotMissing = ~isnan(validationPredictions) & ~isnan(validationResponse);
validationRMSE = sqrt(nansum(( validationPredictions - validationResponse ).^2) / numel(validationResponse(isNotMissing) ));
p=expdata;
p.FoulingFactorm2KkW=[];
p=p{:,:};
y=expdata.FoulingFactorm2KkW;%input
for i=1:7
    X(:,i)=expdata{:,i};
end
for i=1:7
    [mu(:,i),S(:,i),RD(:,i),chi_crt(:,i)]=DetectMultVarOutliers(X(:,i));
end
y=[trainingResponse;validationResponse];%input
yhat=[trainingPredictions;validationPredictions];%response
yhatdash=yhat;%response
r=y-yhat;%residual
for i=1:7
    a(:,i)=0.99*(-1+2*((RD(:,i)-min(RD(:,i)))/(max(RD(:,i))-min(RD(:,i)))));
end
p=expdata;
p.FoulingFactorm2KkW=[];
p=p{:,:};
n=7;
%for i=1:n
 %  w(:,i)=(1-(p(:,i)/(6*m)).^2).^2; 
%end
%% 
z=zscore(yhat,1,'all');
ol=isoutlier(z,"movmedian",3);
oli1=isoutlier(RD(:,1),"movmedian",3);
oli2=isoutlier(RD(:,2),"movmedian",3);
oli3=isoutlier(RD(:,3),"movmedian",3);
oli4=isoutlier(RD(:,4),"movmedian",3);
oli5=isoutlier(RD(:,5),"movmedian",3);
oli6=isoutlier(RD(:,6),"movmedian",3);
oli7=isoutlier(RD(:,7),"movmedian",3);
orgol=ol;
for i=1:size(z)
    if ol(i)==1
        c=1;
         rob(i,c)=z(i-1);
         rob(i,c+1)=z(i);
         rob(i,c+2)=z(i+1);
    end
end
v=0;
for i=1:size(z)
    if ol(i)==1
        v=v+1;
    end
end
percentoutlierinitial=(v/size(z,1))*100;
for i=1:size(z)
    if ol(i)==1
        c=1;
        if abs((rob(i,c)-rob(i,c+1)))<0.1
          ol(i)=0;  
        end
    end
end
for i=1:size(z)
    if oli1(i)==1
        c=1;
         rob1(i,c)=RD(i-1,1);
         rob1(i,c+1)=RD(i,1);
         rob1(i,c+2)=RD(i+1,1);
    end
end
for i=1:size(z)
    if oli2(i)==1
        c=1;
         rob2(i,c)=RD(i-1,2);
         rob2(i,c+1)=RD(i,2);
         rob2(i,c+2)=RD(i+1,2);
    end
end
for i=1:size(z)
    if oli3(i)==1
        c=1;
         rob3(i,c)=RD(i-1,3);
         rob3(i,c+1)=RD(i,3);
         rob3(i,c+2)=RD(i+1,3);
    end
end
for i=1:size(z)
    if oli4(i)==1
        c=1;
         rob4(i,c)=RD(i-1,4);
         rob4(i,c+1)=RD(i,4);
         rob4(i,c+2)=RD(i+1,4);
    end
end
for i=1:size(z)
    if oli5(i)==1
        c=1;
         rob5(i,c)=RD(i-1,5);
         rob5(i,c+1)=RD(i,5);
         rob5(i,c+2)=RD(i+1,5);
    end
end
for i=1:size(z)
    if oli6(i)==1
        c=1;
         rob6(i,c)=RD(i-1,6);
         rob6(i,c+1)=RD(i,6);
         rob6(i,c+2)=RD(i+1,6);
    end
end
for i=1:size(z)
    if oli7(i)==1
        c=1;
         rob7(i,c)=RD(i-1,7);
         rob7(i,c+1)=RD(i,7);
         rob7(i,c+2)=RD(i+1,7);
    end
end
%only case where it can be considered as an outlier is when it is irregular
%in both input and output
    for i=1:size(z)
        if ol(i)==1
            if oli1(i)==1
                p(i,1)=(p(i-1,1)+p(i+1,1))/2;
            end
            if oli2(i)==1
                p(i,2)=(p(i-1,2)+p(i+1,2))/2;
            end
            if oli3(i)==1
                p(i,3)=(p(i-1,3)+p(i+1,3))/2;
            end
            if oli4(i)==1
                p(i,4)=(p(i-1,4)+p(i+1,4))/2;
            end
            if oli5(i)==1
                p(i,5)=(p(i-1,5)+p(i+1,5))/2;
            end
            if oli6(i)==1
                p(i,6)=(p(i-1,6)+p(i+1,6))/2;
            end
            if oli7(i)==1
                p(i,7)=(p(i-1,7)+p(i+1,7))/2;
            end
        end
    end
   
% for i=1:size(z)
%     if ol(i)==1
%         yhat(i)=(yhat(i-1)+yhat(i+1))/2;
%     end
% end
p(:,8)=yhat(:);
T=array2table(p,'VariableNames',{'DensityKgm3', 'Timehr', 'SurfaceTemperatureoC', 'FluidTemperatureoC', 'FluidVelocityms', 'EquivalentDiameterm', 'DissolvedOxygenppmw','FoulingFactorm2KkW'});
[newrmse, yhat1, y1, ol1, percentoutlier]=baggedtree2(T);
gh=0;
hj=0;
for i=1:size(yhat1,1)
    gh=gh+(yhat1(i)-y1(i)).^2;
    hj=hj+(y1(i)-mean(y1(:))).^2;
end
rsq=1-(gh./hj);