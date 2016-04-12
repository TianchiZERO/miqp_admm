% Hybrid vehicle example.

close all
rng('default')
vehicle_data

useADMM = 1;
useGUROBI = 0;

%%%%%%%%%%%%%%%%%%%%%%
% Generating problem %
%%%%%%%%%%%%%%%%%%%%%%
Problem.objective.P = P;
Problem.objective.q = q;
Problem.constraint.A = A;
Problem.constraint.b = b;
Problem.constraint.l1 = l1;
Problem.constraint.l2 = 0;
Problem.constraint.l3 = 0;
Problem.constraint.l4 = l4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solving with ADMM heuristic %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if useADMM
    maxiter = 4000; % max number of iterations
    repeat = 1; % number of initializations
    xs = solver_miqp_admm(Problem, 2, maxiter, repeat);
    
    f_admm = Inf;
    x_admm = zeros(n, 1);
    res = 1e-3;
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

    eng_on  = x_admm(1: T);
    P_eng   = x_admm(end - 4 * T + 2: end - 3 * T + 1);
    P_batt  = x_admm(end - 3 * T + 2: end - 2 * T + 1);
    E_batt  = x_admm(end - 2 * T + 2: end - T + 1);

    % Plotting the results
    figure
    subplot(411)
    plot(0: T, [E_0; E_batt])
    axis([0, 72, 0, E_max])
    subplot(412)
    plot(0: T, [P_batt; 0]);
    axis([0, 72, -2, 2])
    subplot(413)
    plot(0: T, [P_eng; 0]);
    axis([0, 72, 0, 1])
    subplot(414)
    plot(0:T, [eng_on; 0]);
    axis([0, 72, 0, 1])
end

%%%%%%%%%%%%%%%%%%%%%%%
% Solving with Gurobi %
%%%%%%%%%%%%%%%%%%%%%%%
norm(A)

if useGUROBI
    cvx_begin quiet
    cvx_solver gurobi
    cvx_solver_settings('TimeLimit', 20)
        variable x_cvx(n) 
        variable z(l1) binary
        minimize (0.5 * x_cvx' * P * x_cvx + q' * x_cvx + r)
        subject to
            x_cvx(1: l1) == z
            x_cvx(l1 + 1: l1 + l2) >= 0
            A * x_cvx == b
            0 <= z <= 1
    cvx_end
    fprintf('The best value found by Gurobi is %3s\n', cvx_optval);

    eng_on  = x_cvx(1: T);
    P_eng   = x_cvx(end - 4 * T + 2: end - 3 * T + 1);
    P_batt  = x_cvx(end - 3 * T + 2: end - 2 * T + 1);
    E_batt  = x_cvx(end - 2 * T + 2: end - T + 1);

    % Plotting the results
    figure
    subplot(411)
    plot(0: T, [E_0; E_batt])
    axis([0, 72, 0, E_max])
    subplot(412)
    plot(0: T, [P_batt; 0]);
    axis([0, 72, -2, 2])
    subplot(413)
    plot(0: T, [P_eng; 0]);
    axis([0, 72, 0, 1])
    subplot(414)
    plot(0: T, [eng_on; 0]);
    axis([0, 72, 0, 1])
end