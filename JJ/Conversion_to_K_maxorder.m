function [K_maxorder] = Conversion_to_K_maxorder(K_i, modelterms, modelterms_maxorder)

clear K_maxorder
    modelterms_k = modelterms;
    modelterms_k(:,3) = K_i;
    K_maxorder_t = modelterms_maxorder;
    K_maxorder_t(:,3) = 0;
    
    for tt=1:size(modelterms_k,1)
        K_maxorder_t(ismember(K_maxorder_t(:,1:2),modelterms_k(tt,1:2),'rows'),3) = modelterms_k(tt,3);
        K_maxorder_t(ismember(K_maxorder_t(:,1:2),modelterms_k(tt,1:2),'rows'),4) = 1;
    end
    
    K_maxorder(:,1:2) = K_maxorder_t(:,3:4);
    K_maxorder(:,3:4) = K_maxorder_t(:,1:2);
