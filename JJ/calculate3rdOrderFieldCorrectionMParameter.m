function [M_x M_y] = calculate3rdOrderFieldCorrectionMParameter(num_source, modelterms_f_x, modelterms_f_y)


    Xf = num_source(:,7);
    Yf = num_source(:,8);
    
%     modelterms_f_x = [0 0;1 0;0 1;2 0;0 2;3 0;0 3];
%     modelterms_f_y = [0 0;1 0;0 1;2 0;1 1; 0 2;3 0;2 1];
    
    for m = 1:size(modelterms_f_x,1)
        M_x(:,m) = Xf.^modelterms_f_x(m,1) .*Yf.^modelterms_f_x(m,2);
    end
    for m = 1:size(modelterms_f_y,1)
        M_y(:,m) = Yf.^modelterms_f_y(m,1) .*Xf.^modelterms_f_y(m,2);
    end
end
