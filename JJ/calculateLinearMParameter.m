function [M_x M_y] = calculateLinearMParameter (num_source)
    
    M_x = ones([size(num_source,1),1]);
    M_x(:,2:5) = num_source(:,5:8);
    
    M_y = ones([size(num_source,1),1]);
    M_y(:,2:5) = num_source(:,[6 5 8 7]);
    
end
