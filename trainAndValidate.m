function model = trainAndValidate(labelData, featureData)

if(size(labelData,1) ~= size(featureData,1) || size(labelData,2)~=1)
    err('Labels must be col vec with same size as feature data rows')
else

    totalNumExamples = size(labelData,1);

    trainingSizeRatio = 0.75;
    cutoff = floor(totalNumExamples*trainingSizeRatio);
    trainingIndicies = (1:cutoff);
    validationIndicies = (cutoff+1 : totalNumExamples);
    
    model = svmtrain(labelData(trainingIndicies),...
                     featureData(trainingIndicies,:),...
                     '-s 3 -t 2 -p 0.01 -g 0.0001 -h 0 -q');
    disp('TRAINING SET:')
    svmpredict(labelData(trainingIndicies),...
               featureData(trainingIndicies,:),...
               model);
    disp('CROSS VALIDATION SET:')
    svmpredict(labelData(validationIndicies),...
               featureData(validationIndicies,:),...
               model);
end

end