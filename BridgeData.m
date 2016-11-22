classdef BridgeData < handle

    properties
        time;
        speed;
        mass;
        temp;
        humidity;
        naturalFreq;
        critSpeed;
        circularFreq;
        speed2CritSpeed;
        dampingFreq;
        maxDisp;
        varNames = {
            'naturalFreq',...
            'speed',...
            'mass',... 
            'humidity',...
            'critSpeed',...
            'circularFreq',...
            'speed2CritSpeed',...
            'dampingFreq',...
            'maxDisp', ...
            'temp',...
            'time' ...
        };
        varAverages;
        varStdDs;
        rawData;
    end
    methods
        function index = getDataIndex(obj, queryString)
            index = find(strcmp(obj.varNames,queryString));
        end
        
        function res = invZScore(obj, value, varName)
            index = obj.getDataIndex(varName);
            res = value * obj.varStdDs(index) + obj.varAverages(index);
        end
        
        function s = getDataSize(obj, dim)
            s = size(obj.rawData, dim);
        end
        
        function obj = shuffleData(obj)
            ordering = randperm(obj.getDataSize(1));
            obj.rawData = obj.rawData(ordering,:);
        end
        
        function obj = zscoreData(obj)
            [obj.rawData, mu, sig] = zscore(obj.rawData);
            obj.varAverages = mu';
            obj.varStdDs = sig';
        end
        
        function obj = BridgeData(file)
            vars = {'Td','c','dmv','Tact','rh','wn','cr','w','alpha','wb','umax'};
            S = load(file,vars{:});
            obj.time = mod(S.Td',24);
            obj.speed = S.c';
            obj.mass = S.dmv';
            obj.temp = S.Tact';
            obj.humidity = S.rh';
            obj.naturalFreq = S.wn';
            obj.critSpeed = S.cr';
            obj.circularFreq = S.w';
            obj.speed2CritSpeed = S.alpha';
            obj.dampingFreq = S.wb';
            obj.maxDisp = S.umax';
            obj.rawData = [
                obj.naturalFreq ...  
                obj.speed ...
                obj.mass ...
                obj.humidity ...
                obj.critSpeed ...
                obj.circularFreq ...
                obj.speed2CritSpeed ...
                obj.dampingFreq ...
                obj.maxDisp ...
                obj.temp ...
                obj.time ...    
            ];
            numFeatures = size(obj.rawData,2);
            obj.varAverages = zeros(numFeatures,1);
            obj.varStdDs = ones(numFeatures,1);
        end
    end
    

end