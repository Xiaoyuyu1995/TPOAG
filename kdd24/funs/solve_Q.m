function Q_final = solve_Q(Q, L, A, lambda2, beta, maxIter)
% min tr(F^T * L * F) - (1 / lambda2) * tr(F^T * A)

    num_V = length(L);
    num_N = size(L{1}, 1);
    
    for v = 1:num_V
        objQ = [];
        iter = 1;
        Q_old = Q{v};
        while 1
            temp = (beta * eye(num_N) - L{v}) * Q{v} + A{v} ./ (2 * lambda2);
            [U, ~, V] = svd(temp, 'econ');
            Q{v} = U * V';
            objQ = [objQ norm(Q{v} - Q_old, 'fro')];
            if objQ(iter) < 1e-8  || iter > maxIter
                break
            end
            iter = iter + 1;
            Q_old = Q{v};
        end
    end
    Q_final = Q;
end