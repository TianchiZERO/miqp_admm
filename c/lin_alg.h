#ifndef LIN_ALG_H
#define LIN_ALG_H
#define EPS 1e-4

// a matrix in compressed sparse column format 
typedef struct CSC{
    int m, n;
    int *Ap, *Ai;
    double *Ax;
} csc;

// Contains an LDL factorization of symmetric matrix.  
typedef struct LDL_MATRIX{ // contains the factored KKT matrix, and assoc. workspaces
    int n; // size length of the (square) matrix 
    int *Lp, *Li, *Parent, *Lnz, *Flag, *Pattern;
    double *Lx_factor, *Y, *D;
    double *Lx_solve, *D_inv;
    int *p, *pinv;
} ldl_matrix;

// computes the inner product of two vectors
double inner_prod(double *x, double *z, int n);

// prints a matrix as a dense matrix
void print_mat(csc *A, char * name);

//copies z onto x
void vec_copy(double *x, double *z, int n);

// finds the norm of a vector
double vec_norm(double *z, int n);

// updates z1 with z1+z2
void vec_add(double *z1, double *z2, int n);

// updates z1 with z1-z2
void vec_sub(double *z1, double *z2, int n);

//zeros out a vector of length n
void reset_u(double *u, int n);

// l1 is the number of booleans, l2 ternaries, and n the length
void random_init(double *ans, int l1, int l2, int n);

// first l1 are boolean; next l2 are ternary; next l3 are nonnegative
void projection(double *z, double *ans, int l1, int l2, int l3, int n);

// multiplies b[i] by e[i] for every i. Here n is the length.
void entrywise_mult(double *b, double *e, int n);

// find 1./norms(A',1)
void inv_norms_t(csc *As, double *z);

// divides all entries of a vector by norm_factor. (l is the length)
void vec_normalize(double *z, int l, double norm_factor);

// divides all entrie of a matrix by norm_factor
void mat_normalize(csc *A, double norm_factor);

// find the entry with maximum absolute value
double max_abs(csc *As);

// convert a dense matrix to csc form 
// (include_diags set to 1 means always allocate memory to diagonal elements)
void dense_to_csc(int m, int n, double Ad[m][n], csc *As, int include_diags);

// returns z = A*x
void mat_vec(csc *A, double *x, double *z);

// returns z = A'*x
void tr_mat_vec(csc *A, double *x, double *z);

// Premultiply a sparse matrix A by a diagonal matrix given by vector f.  A is overwritten.
void premult_diag(csc *A, double *f);

// Premultiply a sparse matrix A by a diagonal matrix given by vector e.  A is overwritten.
void postmult_diag(csc *A, double *e);

// compare two csc matrices
int compare_mat(csc *A, csc *B);

// invert a permutation matrix, given by permutation vector p.
void invert_p(int *p, int *pinv, int n);

// Allocate the LDL factorization,  which requires computing a symbolic LDL factorization of the KKT matrix.
void alloc_ldl(ldl_matrix *ldl, csc *A, int *p, int *pinv);

// sparse factorization
void factorize_numeric(csc *A, ldl_matrix *ldl);

// return permutation vector p such that diag(p)'*A*diag(p) has approximate minimum degree
void amd(csc *P, int *p);

// solve P*L*D*L'*P'*z = b, return z
void backsolve(ldl_matrix *ldl, double *b, double *z, double *x_work);

// form [A; B]
void vcat(csc *A, csc *B, csc *Z);

// form [A, B]
void hcat(csc *A, csc *B, csc *Z);

// return 0 if vectors are the same, 1 otherwise
int compare_vec(double *x, double *z, int n);

// return the (modified) A and e such that diag(e)*A*diag(e) is equilabrated.
void equilibrate(csc *A, double *e);

void permute(int n,double *x,double *y, int *p);

void ipermute(int n,double *x,double *y, int *p);

// Free memory allocated to an LDL factorization
void free_ldl(ldl_matrix *ldl);

// Free memory allocated to a sparse matrix
void free_csc(csc *A);

// Wrapper function that fills ldl with an LDL factorization, given symmetric matrix A.
void factorize(ldl_matrix *ldl, csc *A);

// adds identity (NOTE: needs memory allocated to digonal elements, otherwise fails silently)
void add_identity(csc *A, double rho);

// returns a multiple of the identity matrix
csc *form_identity(int n, double rho);

// returns A'
csc *transpose(csc *A);

#endif
