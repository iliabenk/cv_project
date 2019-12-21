function [name] =  label_to_name(label)
switch label
    case 1
        name = 'red';
    case 2
        name = 'white';
    case 3
        name = 'blue';
    case 4
        name = 'green';
    case 5
        name = 'orange';
    case 6
        name = 'gray';
    otherwise
        error('Unknown Track label');
end