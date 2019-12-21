function [optimal] = find_optimal_number(img)

listOfMona = {'Mona1_Y.jpg','Mona2_Y.jpg','Mona3_Y.jpg','Mona4_Y.jpg','Mona5_N.jpg',...
    'Mona6_N.jpg','Mona7_Y.jpg','Mona8_Y.jpg','Mona9_Y.jpg','Mona10_Y.jpg'};

listOfAnswers = [1 1 1 1 0 0 1 1 1 1];
numOfSuccesses = [];
for num = 1 : 26
    
    straightMonaPoints = detectSURFFeatures(img,'NumOctaves' , 5 , 'NumScaleLevels' , 7 );
    [straightMonaFeatures , straightMonaValidPoints] = extractFeatures(img , straightMonaPoints.selectStrongest(num));
    
    numOfCorrectChoices = 0;
    
    for monaIndex = 1 : length(listOfMona)
        monaImg = imread(listOfMona{monaIndex});
        monaImg = double(monaImg) ./ 255;
        monaImg = rgb2gray(monaImg);
        
        monaPoints = detectSURFFeatures(monaImg);
        [monaFeatures , monaValidPoints] = extractFeatures(monaImg , monaPoints);
        
        monaPairs = matchFeatures(straightMonaFeatures , monaFeatures , 'MatchThreshold' , 1.7);
        
        if (size(monaPairs , 1) >= 1)
            success = 1;
        else
            success = 0;
        end
        
        if success == listOfAnswers(monaIndex)
            numOfCorrectChoices = numOfCorrectChoices + 1;
        end
    end
    
    numOfSuccesses = [numOfSuccesses , numOfCorrectChoices];
    
end
% optimal = max(numOfSuccesses);
optimal = [7 8 9 10];
majority(img , optimal);

function [junk] =  majority(img, optimal)
listOfMona = {'Mona1_Y.jpg','Mona2_Y.jpg','Mona3_Y.jpg','Mona4_Y.jpg','Mona5_N.jpg',...
    'Mona6_N.jpg','Mona7_Y.jpg','Mona8_Y.jpg','Mona9_Y.jpg','Mona10_Y.jpg'};

listOfAnswers = [1 1 1 1 0 0 1 1 1 1];
numOfSuccesses = [];
answers = zeros(4,10);

for num = 1 : length(optimal)
    
    straightMonaPoints = detectSURFFeatures(img);
    [straightMonaFeatures , straightMonaValidPoints] = extractFeatures(img , straightMonaPoints.selectStrongest(optimal(num)));
    
    numOfCorrectChoices = 0;
    
    for monaIndex = 1 : length(listOfMona)
        monaImg = imread(listOfMona{monaIndex});
        monaImg = double(monaImg) ./ 255;
        monaImg = rgb2gray(monaImg);
        
        monaPoints = detectSURFFeatures(monaImg);
        [monaFeatures , monaValidPoints] = extractFeatures(monaImg , monaPoints);
        
        monaPairs = matchFeatures(straightMonaFeatures , monaFeatures);
        
        answers(num , monaIndex) = (size(monaPairs , 1) >= 1);
        
    end
    
    
    
end

answers2 = sum(answers , 1);
answers2 = (answers2 >= 2);

numOfSuccess = sum(answers2 == listOfAnswers);

numOfSuccess


