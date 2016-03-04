close all;
% fuel use is given by F(p) = p+ gamma*p^2 (for p>=0)

% define Preq, required power at wheels
% Preq is piecewise linear
% a is slope of each piece
a=[0.5 -0.5 0.2 -0.7 0.6 -0.2 0.7 -0.5 0.8 -0.4];
% l is length of each piece
l=[40 20 40 40 20 40 30 40 30 60];

Preq=(a(1):a(1):a(1)*l(1))';
for i=2:length(l)
    Preq=[Preq; (Preq(end)+a(i):a(i):Preq(end)+a(i)*l(i))'];
end


% MODEL
P_des = Preq(1:5:end) * 2;
T = length(P_des);
tau = 350 / length(P_des);
P_eng_max = 1;
alpha = .2*tau;
beta = 2*tau;
gamma = 30*tau*.01;
f_tilde = @(P_eng, eng_on) alpha*square(P_eng) + beta*P_eng + gamma*eng_on;
delta = 1000*.01;
eta = .1;
E_max = 2e2;
E_0 = E_max;

% BUILD MATRICES; vector is [P_eng(T); P_batt(T); E_batt(T); turn_on(T-1), eng_on(T)];
A = [];
b = [];

% dynamics
A = [zeros(T-1,T), ...
     toeplitz([0,zeros(1,T-2)], [0,1,zeros(1,T-2)]), ...
     toeplitz([-1,zeros(1,T-2)], [-1,1,zeros(1,T-2)]) / tau, ...
     zeros(T-1,T-1), ...
     zeros(T-1,T)];
b = zeros(T-1,1);

% initial dynamics
A = [A; [zeros(1,T), tau, zeros(1,T-1), 1, zeros(1,T-1), zeros(1,T-1), zeros(1,T)]]; 
b = [b; E_0];

% power balance
G = [eye(T), eye(T), zeros(T, 3*T-1)];
h = P_des;

% battery limits
G = [G; [zeros(T, 2*T), eye(T), zeros(T, 2*T-1)]];
h = [h; zeros(T,1)];
G = [G; [zeros(T, 2*T), -eye(T), zeros(T, 2*T-1)]];
h = [h; -E_max*ones(T,1)];

% P_eng limits
G = [G; [eye(T), zeros(T, 3*T-1), zeros(T)]];
h = [h; zeros(T,1)];
G = [G; [-eye(T), zeros(T, 3*T-1), P_eng_max*eye(T)]];
h = [h; zeros(T,1)];

% turn_on
G = [G; [zeros(T, 3*T), eye(T, T-1), zeros(T)]];
h = [h; zeros(T,1)];
G = [G; [zeros(T-1,3*T), ...
     eye(T-1,T-1) ...
     -toeplitz([-1,zeros(1,T-2)], [-1,1,zeros(1,T-2)])]];
h = [h; zeros(T-1,1)];

% fuel cost
Phalf = [ ... 
          sqrt(alpha)*eye(T), zeros(T, 4*T-1)
          zeros(1, 3*T-1), sqrt(eta), zeros(1,T-1 + T)
        ];
P = Phalf'*Phalf;
P = 2*P; % objective will be (1/2)x^TPx, not x^TPx
q = [beta*ones(T, 1); zeros(4*T-1, 1)];
q = q + [zeros(3*T-1,1); -2*eta*E_max; zeros(T-1 + T, 1)];
q = q + [zeros(4*T-1,1); gamma*ones(T, 1)];
r = eta*E_max^2;

% turn-on cost
q = q + delta*[zeros(3*T,1); ones(T-1, 1); zeros(T, 1)];


%
% put problem in standard form with slack variables
l = length(h);
k = length(P);
P = [P, zeros(k,l); zeros(l,k), zeros(l,l)];
q = [q; zeros(l,1)];
A = [A, zeros(size(A,1),l); G, -eye(l)];
b = [b; h];

% permute vectors (our standard form isn't the form I wrote the matrices... should change later)
n1 = 4*T-1;
n2 = T;
n3 = l;
n = n1 + n2 + n3;
Perm = [ ...
          zeros(n1,n2), zeros(n1,n3),      eye(n1),
               eye(n2), zeros(n2,n3), zeros(n2,n1),
          zeros(n3,n2),      eye(n3), zeros(n3,n1),
       ];
   
% normalizing
A = 100*A*Perm;
P = 100*Perm'*P*Perm;
q = Perm'*q;

l1 = n2;
l3 = n3;
l2 = 0;
%l3 = n1;

