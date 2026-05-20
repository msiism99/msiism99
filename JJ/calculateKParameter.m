function [fitted residual] = calculateFittedResidual(num_source, M_x, M_y, K_x, K_y)    
    fitted = num_source;
    residual = num_source;
    
    fitted(:,9) = M_x * K_x;
    fitted(:,10) = M_y * K_y;
    
    residual(:,9:10) = num_source(:,9:10) - fitted(:,9:10);
end
