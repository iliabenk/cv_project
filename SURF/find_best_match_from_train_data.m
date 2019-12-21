function [label, matched_points] = find_best_match_from_train_data(train_data, I_test_features, I_test_vaild_points, pairs_thr)

    best_match_index = 0;
    best_match_num_of_pairs = 0;
    match_hist = zeros(6, 1);
    valid_data = train_data;
    valid_img_pairs_index = 1;
    is_found_any_matching_train_image = 0;
    
    while 0 == is_found_any_matching_train_image
        for train_index = 1 : numel(train_data)
            I_train_features = train_data(train_index).features;
            I_train_valid_points = train_data(train_index).points;

            img_pairs = matchFeatures(I_train_features , I_test_features , 'MatchThreshold' , 1.7);

            num_of_pairs = size(img_pairs, 1);

            if num_of_pairs >= pairs_thr
               is_found_any_matching_train_image = 1;
               match_hist(train_data(train_index).label) = match_hist(train_data(train_index).label) + 1;   
               valid_img_pairs(valid_img_pairs_index).pairs = img_pairs;
               valid_img_pairs(valid_img_pairs_index).label = train_data(train_index).label;
               valid_img_pairs_index = valid_img_pairs_index + 1;
            end

    %         if num_of_pairs > best_match_num_of_pairs
    %            best_match_num_of_pairs = num_of_pairs;
    %            best_match_index = train_index;
    %            best_img_pairs = img_pairs;
    %         end
        end
        
        pairs_thr = pairs_thr - 1;
    end
    
    [~, label] = max(match_hist);
    
    valid_img_pairs = remove_wrong_labels_from_data(valid_img_pairs, label);
    
    valid_test_img_points_after_reduction = get_match_points_after_choosing_label(valid_img_pairs);
    
    matched_points = I_test_vaild_points(valid_test_img_points_after_reduction);

end