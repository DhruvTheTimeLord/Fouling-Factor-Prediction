function [ newRMSE, yhat1, y1, ol1, percentoutlier] = baggedtree2(T)
inputTable = T;
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
inputTable = T;
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
yhat1=[validationPredictions];
y1=[validationResponse];%input
newRMSE = sqrt(nansum(( yhat1-y1 ).^2) / numel(yhat1(isNotMissing) ));
z1=zscore(yhat1,1,'all');
ol1=isoutlier(z1,"movmedian",3);
for i=1:size(z1)
    if ol1(i)==1
        c=1;
         rob1(i,c)=z1(i-1);
         rob1(i,c+1)=z1(i);
         rob1(i,c+2)=z1(i+1);
    end
end
for i=1:size(z1)
    if ol1(i)==1
        c=1;
        if abs(abs(rob1(i,c))-abs(rob1(i,c+1)))<0.1
          ol1(i)=0;  
        end
    end
end
v=0;
for i=1:size(z1)
    if ol1(i)==1
        v=v+1;
    end
end
percentoutlier=(v/size(z1,1))*100;
for i=1:size(z1)
    if ol1(i)==1
        yhat1(i)=(yhat1(i-1)+yhat1(i+1))/2;
    end
end
newRMSE = sqrt(nansum(( yhat1-y1 ).^2) / numel(yhat1(isNotMissing) ));
yhat1=[trainingPredictions;validationPredictions];
y1=[trainingResponse;validationResponse];%input
