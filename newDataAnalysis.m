
clear;
fileName = 'newdata.mat';
bData = BridgeData(fileName);
iVarName = 'time';
dVarName = 'temp';

iVarIndex = bData.getDataIndex(iVarName);
dVarIndex = bData.getDataIndex(dVarName);
%get our graph
%trainAndCompare(independentVar, dependentVar, bDdata);

bData.shuffleData()
bData.zscoreData();

labelData = bData.rawData(:,dVarIndex);
featureData = bData.rawData(:,setdiff(1:bData.getDataSize(2),dVarIndex));

model = trainAndValidate(labelData, featureData);

X = (-3:0.1:3)';
visRawData = zeros(size(X,1),bData.getDataSize(2));
visRawData(:,iVarIndex) = X;

visLabel = visRawData(:,dVarIndex);
visFeature = visRawData(:,setdiff(1:bData.getDataSize(2),dVarIndex));

Y = svmpredict(visLabel, visFeature, model, '-q');

X = bData.invZScore(X,iVarName);
Y = bData.invZScore(Y,dVarName);


figure
plot(X,Y);
xlabel(iVarName);
ylabel(dVarName);