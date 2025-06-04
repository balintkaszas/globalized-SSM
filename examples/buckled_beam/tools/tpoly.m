
%% Subfunction: Express tensor as polynomial function
function xi = tpoly(T, z)
    xi = zeros(size(T(1).coeffs,1), size(z, 2));
    z = reshape(z, 1, size(z, 1), []);
    for o = 1:numel(T)
        if ~isempty(T(o).ind)
            xi = xi + T(o).coeffs*reshape(prod(z.^T(o).ind, 2), size(T(o).ind, 1), []);
        end
    end
end