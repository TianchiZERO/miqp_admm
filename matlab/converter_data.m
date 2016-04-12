% Model
L1 = 1e-5;
C1 = 1e-6;
L2 = 1e-5;
C2 = 1e-5;
R = 1;
Vdc = 10;
T = 100;
tau = 5e-7;
regularization = .1;
sw_cost = 1.5;

Gtilde = [
            0,  -1/L1,      0,     -1/L1
         1/C1,      0,  -1/C1,         0
            0,   1/L2,      0,         0
         1/C2,      0,      0, -1/(R*C2)
     ];
htilde = [Vdc/L1; 0; 0; 0];

% Discretize
G = expm(Gtilde * tau);
h = Gtilde \ (G - eye(size(Gtilde))) * htilde;
n_xi = length(h);

% Formulate objective
v_des = sin(2 * pi * (1: T) / T);
C = [0, 0, 0, 1];
omega = 2 * pi / (T * tau);
H = C / (Gtilde - 1j * omega * eye(4)) * [1; 0; 0; 0];
v_in = max(v_des) / H;
x_ss = (Gtilde - 1j * omega * eye(4)) \ [1; 0; 0; 0] * v_in;
x_ss = real(x_ss) * sin(omega * (1: T) * tau) + imag(x_ss) * cos(omega * (1: T) * tau);

P_xi = C' * C + regularization * eye(n_xi);
for t = 1:T
  q_xi(1: n_xi,t) = -2 * C' * v_des(t) - 2 * regularization * x_ss(:, t);
  r_xi(t) = v_des(t)' * v_des(t) + regularization * x_ss(:, t)' * x_ss(:, t);
end

% Form matrices (vector is xi(n_xi * (T + 1)), u(m * (T + 1)), turn_on(m * T) )
A = [];
b = [];

% Dynamics
A = [A; kron(eye(T, T + 1), G) - kron([diag(ones(T - 1, 1), 1),  ...
       [zeros(T - 1, 1); 1]], eye(size(G))), kron(eye(T, T + 1), h), ...
       zeros(T * n_xi, T)];
b = [b; zeros(T * n_xi, 1)];

% Periodicity
A = [A; eye(n_xi), zeros(n_xi, (T - 1) * n_xi), -eye(n_xi), zeros(n_xi, T + 1), zeros(n_xi, T)];
b = [b; zeros(n_xi, 1)];
A = [A; zeros(1, (T + 1) * n_xi), 1, zeros(1, T - 1), -1, zeros(1, T)];
b = [b; 0];

% Inequalities
G = [];
h = [];

% Turn on
D = eye(T, T + 1) - [zeros(T, 1), eye(T)];
G = [G; zeros(T, (T + 1) * n_xi), kron(D, 1), eye(T)];
h = [h; zeros(T, 1)];

% Tracking objective
P = zeros(n_xi * (T + 1) + 2 * T + 1);
P(1: T * n_xi, 1: T * n_xi) = kron(eye(T), P_xi);
q = zeros(n_xi, 1);
for t = 0:T-1
  q = [q; q_xi(:, t + 1)];
end
q = [q; zeros(T + 1, 1); zeros(T, 1)];
r = sum(r_xi);

% Switch cost
q = q + [zeros((T + 1) * n_xi, 1); zeros(T + 1, 1); sw_cost * ones(T, 1)];

% Put problem in standard form with slack variables
l = length(h);
k = length(P);
P = [P, zeros(k, l); zeros(l, k), zeros(l)];
P = 2*P;
q = [q; zeros(l, 1)];
A = [A, zeros(size(A, 1), l); G, -eye(l)];
b = [b; h];

% Permute vectors to the standard form
l2 = T + 1;
l4 = l + T; % size(G,1) + length(turn_on)
l5 = n_xi * (T + 1);
n = l2 + l4 + l5;
Perm = [ ...
          zeros(l5, l2), zeros(l5, l4),       eye(l5),
                eye(l2), zeros(l2, l4), zeros(l2, l5),
          zeros(l4, l2),       eye(l4), zeros(l4, l5),
       ];
A = A * Perm;
P = Perm' * P * Perm;
q = Perm' * q;

