function [M_x M_y] = calculate3rdOrderWaferCorrectionMParameter(num_source, modelterms_w_x, modelterms_w_y)

    Xw = num_source(:,5);
    Yw = num_source(:,6);
    
    for m = 1:size(modelterms_w_x,1)
        M_x(:,m) = Xw.^modelterms_w_x(m,1) .*Yw.^modelterms_w_x(m,2);
    end
    
    for m = 1:size(modelterms_w_y,1)
        M_y(:,m) = Yw.^modelterms_w_y(m,1) .*Xw.^modelterms_w_y(m,2);
    end
end
