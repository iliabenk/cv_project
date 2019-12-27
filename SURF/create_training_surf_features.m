function [train_features_labels] = create_training_surf_features(test_dir_path, num_octaves, num_scales)
    list_dir = dir(test_dir_path);
    
    output_index = 1;
    counter_amount_training_per_label = zeros(6, 1);
    
    for file_num = 1 : length(list_dir)
       file_name = list_dir(file_num).name;

       if strcmp(file_name(1), '.') || ~strcmp(file_name(end-3 : end), '.png')
           continue
       end
       
       img_path = fullfile(test_dir_path, file_name);
       
       if contains(file_name, 'red')
           label = 1;
       elseif contains(file_name, 'white')
           label = 2;
       elseif contains(file_name, 'blue')
           label = 3;     
       elseif contains(file_name, 'green')
            label = 4;
       elseif contains(file_name, 'orange')
            label = 5;    
       elseif contains(file_name, 'gray')
            label = 6;
       else
           error(['Unknown color on file: ', file_name]);
       end
       
       counter_amount_training_per_label(label) = counter_amount_training_per_label(label) + 1;
       
       I_train = imread(img_path);
       I_train = double(I_train) ./ 255;
       I_train = rgb2gray(I_train);
%        I_train = imresize(I_train, [800, 600]); %test toresize image
%        I_train = imgaussfilt(I_train, 0.5);
       
       I_train_points = detectSURFFeatures(I_train, 'NumOctaves', num_octaves, 'NumScaleLevels', num_scales);
       [I_train_features , I_train_vaild_points] = extractFeatures(I_train , I_train_points);
    
       train_features_labels(output_index).features = I_train_features;
       train_features_labels(output_index).points = I_train_vaild_points;
       train_features_labels(output_index).label = label;
       
       output_index = output_index + 1;
    end
    
    disp('Total amount of training data per label is:');
    
    for label = 1 : 6
       disp(['num of ', label_to_name(label), ' data images = ', num2str(counter_amount_training_per_label(label))]); 
    end
end
