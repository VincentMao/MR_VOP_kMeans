%% Find if there exists Q+Z-B is psd
function [ Z ] = FindPSD(B_current, B_core, currentZ, Nc)
    % disp 'Finding the PSD matrix Z'; 
    Z = zeros(Nc, Nc);
    maxInters = 300;
    B_core = B_core + currentZ;
    for ii = 1: maxInters
        % printf('%d / %d', ii, maxInters);
        alleigs = eig(B_core+Z-B_current);
        if (all(alleigs >=0) || all(-real(alleigs(alleigs<0)) <= 1e-5) )
            Z = currentZ+Z;
            break;
        else
            % Find the next possible Z
            [V, D] = eig(B_core+Z-B_current);
            D_orig = D;
            D(D<=0)=0;
            D_new = D - D_orig;
            B_new = V*D_new*V';
            Z = Z + B_new;
        end 
    end
end