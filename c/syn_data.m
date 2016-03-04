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
n = 600; % number of variables
m = n/4; % number of constraints

P = full(sprandsym(n, .1, (1:n)/n));
q = randn(n,1);
r = 0.5*q'*(P\q);

l1 = n/2; % number of Boolean variables
l3 = n/4; % number of nonnegative variables

A=randn(m,n);
x0 = [binornd(1,.5,l1,1);1*rand(n-l1,1)];
b = A * x0;

l2 = 0;

if useGUROBI
    cvx_begin
    cvx_solver Gurobi
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
