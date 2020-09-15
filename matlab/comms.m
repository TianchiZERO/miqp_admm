% Communication example.aaa

close all

% Parameter values
n = 200;
m = 5 * n;
K = 4;
noise_level = 25;
diff_val = [];
diff_err = [];

% Encode and decode functions
encode = @(s) 2*s - 3;
decode = @(x) max(min(round((x + 3)/2), 3), 0);
    

for instance = 1:100
    rng(instance)
    x_base = 2 * (floor(4 * rand(n, 1)) - 1.5) + 2j * (floor(4 * rand(n, 1)) - 1.5);
    s = encode(0: 3)';
    H = randn(m, n) + 1j * randn(m, n);
    v = noise_level * (randn(m, 1) + 1j * randn(m, 1));
    y = H * x_base + v;

    Hr = nan(2 * m, 2 * n);
    yr = nan(2 * m, 1);
    sr = nan(2 * K, 1);
    for i = 1:m
      for j = 1:n
        Hr(2 * i - 1: 2 * i, 2 * j - 1: 2 * j) = [real(H(i, j)), -imag(H(i, j)); imag(H(i, j)), real(H(i, j))];
      end
      yr(2 * i - 1: 2 * i) = [real(y(i)), imag(y(i))];
    end
    P = 2 * (Hr' * Hr);
    q = -2 * Hr' * yr;
    r = yr' * yr;

    A = zeros(0, 2 * n);
    b = zeros(0, 1);
    l3 = 2 * n;
    
    % Defining the problem
    Problem.objective.P = P;
    Problem.objective.q = q;
    Problem.constraint.A = A;
    Problem.constraint.b = b;
    Problem.constraint.l1 = 0;
    Problem.constraint.l2 = 0;
    Problem.constraint.l3 = l3;
    Problem.constraint.l4 = 0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Solving with ADMM heuristic %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    maxiter=  20; % max number of iterations
    repeat = 1; % number of initializations
    xs = solver_miqp_admm(Problem, .5, maxiter, repeat);

    f_admm = Inf;
    x_admm = zeros(n, 1);
    
    for j=1:repeat
        for k=1:maxiter
            x = xs(:, (j - 1) * maxiter + k);
            f = 0.5 * x' * P * x + q' * x + r;
            if f < f_admm
                f_admm = f;
                x_admm = x;
            end
        end
    end

    % Calculating the error rate    
    correct = 0;
    for i = 1:n
      if norm(x_admm(2 * i - 1: 2 * i) - [real(x_base(i)); imag(x_base(i))]) <= .1
        correct = correct + 1;
      end
    end
    error_rate_admm = (n - correct) / n;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Relax and round heuristic %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x_rlx = - P \ q;
    x_rlx = sign(x_rlx) .* (2 * (abs(x_rlx) > 2) + 1);
    f_rlx = 0.5 * x_rlx' * P * x_rlx + q' * x_rlx + r;

    % Calculating the error rate    
    correct = 0;
    for i = 1:n
      if norm(x_rlx(2 * i - 1: 2 * i) - [real(x_base(i)); imag(x_base(i))]) <= .1
        correct = correct + 1;
      end
    end
    error_rate_rlx = (n - correct) / n;

    diff_val = [diff_val, f_rlx - f_admm];
    diff_err = [diff_err, error_rate_rlx - error_rate_admm];
end

%%%%%%%%%%%%%%%%%%%
% Plot histograms %
%%%%%%%%%%%%%%%%%%%
subplot(211)
hist(diff_val)

subplot(212)
hist(diff_err)
