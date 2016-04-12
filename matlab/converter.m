% Converter example. 

close all
rng('default')
converter_data

useADMM = 1;
useGurobi = 0;

%%%%%%%%%%%%%%%%%%%%%%
% Generating problem %
%%%%%%%%%%%%%%%%%%%%%%
Problem.objective.P = P;
Problem.objective.q = q;
Problem.constraint.A = A;
Problem.constraint.b = b;
Problem.constraint.l1 = 0;
Problem.constraint.l2 = l2;
Problem.constraint.l3 = 0;
Problem.constraint.l4 = l4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solving with ADMM heuristic %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if useADMM,
    maxiter=7000; % max number of iterations
    repeat = 1; % number of initializations
    xs = solver_miqp_admm(Problem, 4, maxiter, repeat);
   
    f_admm = Inf;
    x_admm = zeros(n,1);
    res= 2e-4;
    normA = norm(A);
    
    for j=1:repeat
        for k=1:maxiter
            x = xs(:, (j - 1) * maxiter + k);
            if norm(A * x - b) / normA < res
                f = 0.5 * x' * P * x + q' * x + r;
                if f < f_admm
                    f_admm = f;
                    x_admm = x;
                end
            end
        end
    end

    assert(f_admm < Inf);
    fprintf('The best value found by ADMM is %3s\n', f_admm);

    % Recover original variables
    u_std = x_admm(1: T + 1);
    xi_std = reshape(x_admm(end - n_xi * (T + 1) + 1 : end), n_xi, T + 1);

    % Plots
    figure
    subplot(211)
    plot(0: T, u_std)
    axis([0, T, -1, 1])
    ylabel('u')
    subplot(212)
    plot(1: T, C * xi_std(:, 2: 101), 1: T, v_des)
    axis([0, T, -1.5, 1.5])
    ylabel('v_{out}')
end

%%%%%%%%%%%%%%%%%%%%%%%
% Solving with Gurobi %
%%%%%%%%%%%%%%%%%%%%%%%
if useGurobi
    cvx_begin quiet
    cvx_solver gurobi
    cvx_solver_settings('TimeLimit', 20)
        variable x_cvx(n) 
        variable z(l2) integer
        minimize (1/2 * x_cvx' * P * x_cvx + q' * x_cvx + r)
        subject to
            x_cvx(1: l2) == z
            x_cvx(l2 + 1: l2 + l4) >= 0
            A * x_cvx == b
            -1 <= z <= 1
    cvx_end
    fprintf('The best value found by Gurobi is %3s\n', cvx_optval);

    % Recover the original variables
    u_std = x_cvx(1: T + 1);
    xi_std = reshape(x_cvx(end - n_xi * (T + 1) + 1 : end), n_xi, T + 1);

    % Plots
    figure
    subplot(211)
    plot(0: T, u_std)
    axis([0, T, -1, 1])
    ylabel('u')
    subplot(212)
    plot(1: T, C * xi_std(:, 2: 101), 1: T, v_des)
    axis([0, T, -1.5, 1.5])
    ylabel('v_{out}')
end