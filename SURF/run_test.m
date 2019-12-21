function [test_features_labels] = run_test(test_dir_path, num_octaves, num_scales, train_data, pairs_thr)
    list_dir = dir(test_dir_path);
    
    output_index = 1;
    
    for file_num = 1 : length(list_dir)
       file_name = list_dir(file_num).name;

       if strcmp(file_name(1), '.') || ~strcmp(file_name(end-3 : end), '.JPG')
           continue
       end
       
       img_path = fullfile(test_dir_path, file_name);
       
       tic
       
       I_test_colored = imread(img_path);
       I_test = double(I_test_colored) ./ 255;
       I_test = rgb2gray(I_test);
%        I_test = imresize(I_test, [800, 600]); %test toresize image
       
       I_test_points = detectSURFFeatures(I_test, 'NumOctaves', num_octaves, 'NumScaleLevels', num_scales);
       [I_test_features , I_test_vaild_points] = extractFeatures(I_test , I_test_points);
    
       test_features_labels(output_index).features = I_test_features;
       test_features_labels(output_index).points = I_test_vaild_points;
       [test_features_labels(output_index).label, test_features_labels(output_index).match_points] = find_best_match_from_train_data(train_data, I_test_features, I_test_vaild_points, pairs_thr);
       test_features_labels(output_index).file_name = file_name;
        
       if pairs_thr
          match_points_location = round(test_features_labels(output_index).match_points.Location);
          
          min_inliers_percent = 0.5;
          data_type = 'points';
          
          %remove outliers before boxing
          match_points_location_cleaned = match_points_location; %in case we do not remove outliers
          match_points_location_cleaned = remove_outliers(match_points_location, min_inliers_percent, data_type);
          
          toc
          
          box_bias_x = 0;
          box_bias_y = 0;
          
          min_row = min(match_points_location_cleaned(:, 1)) - box_bias_x;
          min_col = min(match_points_location_cleaned(:, 2)) - box_bias_y;
          max_row = max(match_points_location_cleaned(:, 1)) + box_bias_x;
          max_col = max(match_points_location_cleaned(:, 2)) + box_bias_y;
          
          figure()
          imshow(I_test_colored);
          hold on;
          
          box = [min_row, min_col; max_row, min_col; max_row, max_col; min_row, max_col; min_row, min_col];

          plot(box(:, 1), box(:, 2), 'LineWidth',3);
          
          title(['Track color : ', label_to_name(test_features_labels(output_index).label)]);
          
          hold off
       end
       
       output_index = output_index + 1;
    end
end
