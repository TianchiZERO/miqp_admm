% Generate some random instances of mixed integer quadratic programs,
% and approximately solve them via ADMM heuristic. For comparison,
% Gurobi can be run on the same problem by useGurobi = 1.

close all
rng('default')

useADMM = 1;
useGUROBI = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generating the problem data %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = 200; % number of variables
m = n/4; % number of constraints

P = randn(n,n); P = P'*P; P = (P+P')/2;
q = randn(n,1);
r = 0.5*q'*(P\q);

l1 = n/2; % number of Boolean variables
l4 = n/4; % number of nonnegative variables

A=randn(m,n);
x0 = [binornd(1,.5,l1,1);1*rand(n-l1,1)];
b = A * x0;

Problem.objective.P = P;
Problem.objective.q = q;
Problem.objective.r = r;
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
    maxiter = 500; % max number of iterations
    repeat = 10; % number of initializations
    xs = solver_miqp_admm(Problem, .5, maxiter, repeat);

    f_admm = Inf;
    x_admm = zeros(n,1);
    res_thr = 1e-4; % feasibilty threshold 

    for j=1:repeat
        for k=1:maxiter
            x = xs(:,(j-1)*maxiter+k);
            res = norm(A*x-b);
            if res < res_thr
                f = 0.5*x'*P*x + q'*x + r;
                if f < f_admm
                    f_admm = f;
                    x_admm = x;
                end
            end
        end
    end

    assert(f_admm < Inf);
    fprintf('The best value found by ADMM is %3s\n', f_admm);
end

%%%%%%%%%%%%%%%%%%%%%%%
% Solving with Gurobi %
%%%%%%%%%%%%%%%%%%%%%%%
if useGUROBI
    cvx_begin quiet
    cvx_solver gurobi
    cvx_solver_settings('TimeLimit', 5)
        variable x_cvx(n) 
        variable z(l1) binary
        minimize (0.5*x_cvx'*P*x_cvx + q'*x_cvx + r)
        subject to
            A*x_cvx == b
            x_cvx(1:l1) == z
            x_cvx(l1+1:l1+l4) >= 0
    cvx_end
    fprintf('The best value found by Gurobi is %3s\n', cvx_optval);
end