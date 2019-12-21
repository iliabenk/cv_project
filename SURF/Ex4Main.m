clear all
close all

figNum = 1;

train_dir = '/Users/iliabenkovitch/Documents/Computer_Vision/git/cv_project/SURF/data/training';
num_octaves = 3;
num_scales = 6;
train_data = create_training_surf_features(train_dir, num_octaves, num_scales);

test_dir = '/Users/iliabenkovitch/Documents/Computer_Vision/git/cv_project/SURF/data/test';
num_octaves = 3;
num_scales = 6;
pairs_thr = 30;

test_data = run_test(test_dir, num_octaves, num_scales, train_data, pairs_thr);

% I_test = imread('/Users/iliabenkovitch/Documents/Computer_Vision/git/cv_project/SURF/data/test/test_1.JPG');
% I_test = double(I_test) ./ 255;
% I_test = rgb2gray(I_test);
% 
% % figure(figNum)
% % figNum = figNum + 1;
% % imshow(I_train);
% % hold on;
% % plot(I_train_vaild_points);
% 
% 
% I_test_points = detectSURFFeatures(I_test,'NumOctaves' , 5 , 'NumScaleLevels' , 7 );
% tic
% [I_test_features , I_test_valid_points] = extractFeatures(I_test , I_test_points);
% toc
% 
% tic
% img_pairs = matchFeatures(I_train_features , I_test_features , 'MatchThreshold' , 1.7);
%    toc 
% if size(img_pairs , 1) >= 1
%     figure(figNum) , imshow(I_test);
%     figNum = figNum + 1;
% 
%     matchedPoints1 = I_train_vaild_points(img_pairs(:, 1));
%     matchedPoints2 = I_test_valid_points(img_pairs(:, 2));
% 
%     figure(figNum);
%     figNum = figNum + 1; 
%     showMatchedFeatures(I_train ,I_test ,matchedPoints1 ,matchedPoints2 ,'montage','Parent',axes);
%     title('Candidate point matches');
% end
%     


