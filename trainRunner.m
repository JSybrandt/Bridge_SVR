% a = acceleration data
% amax = max acceleration data
% Td = time of day (counting from midnight - counts up to 48 to represent 2 days)
% n = number of vehicles on bridge
% Tact = actual temp at time of measurement
% rh = relative humidity at time of measurement 
% m= total mass of bridge
% k = total stiffness of bridge
% wn = natural frequency of bridge considering temp, humidity, and car effects
% e = damping ratio
% c= damping coefficient
% t = time of dynamic measurement (time cars are on bridge)
% u= displacement data
% v = velocity data
% umax = max displacements
% vmax = max velocities

clear;

load('totalData.mat');

fixedTd = mod(Td,24);

data = [amax' fixedTd' n' Tact' rh' m' k' wn' e' c' u' v'];

[data,mu,sig] = zscore(data);
mu = mu';
sig = sig';

%data = data + ((rand(size(data))*2 - 1) .* (rand(size(data)) < .01));

origNumFeatures = size(data,2)-1;
accPerFeature = zeros(3,origNumFeatures);
% for i = 1:origNumFeatures

numPoints = size(data,1);

numFeatures = size(data,2) - 1; 

ordering = randperm(numPoints);

data = data(ordering,:);



trainingSize = floor(numPoints * 0.75);
validationSize = numPoints - trainingSize;

trainingData = data(1:trainingSize,2:end);
validationData = data(trainingSize+1:end,2:end);

trainingValues = data(1:trainingSize,1);
validationValues = data(trainingSize+1:end,1);


model = svmtrain(trainingValues,trainingData, '-s 3 -t 2 -p 0.01 -g 0.0001 -h 0 -q');

disp('TRAINING SET:')
trainingPredictions = svmpredict(trainingValues,trainingData,model);
disp('CROSS VALIDATION SET:')
[validationPredictions, acc, decVals] = svmpredict(validationValues, validationData,model);

accPerFeature(:,i) = acc;

% % %cout out feature #i
% % data = [data(:,1) data(:,3:end)];
% % 
% % size(data)
% % 
% % end
% % 
% % res = zeros(3,origNumFeatures);
% % for i = 1 : origNumFeatures - 1 
% %     res(:,i) = accPerFeature(:,i) - accPerFeature(:,i+1);
% % end
% % res(origNumFeatures) = accPerFeature(:,origNumFeatures);

X = (-3:0.1:3)';
testSize = size(X,1);

cRange = 2;

figure
xlabel('vals -3 to 3');
ylabel('max accel');

for i = cRange
    
    Z = zeros(testSize, numFeatures);
    Z(:,i) = X;
    
    Y = svmpredict(zeros(testSize,1),Z,model,  '-q');
    Yt = invZScore(Y, mu(cRange), sig(cRange));
    Xt = invZScore(X, mu(cRange), sig(cRange));
    plot(Xt,Yt);
    hold on;
end

