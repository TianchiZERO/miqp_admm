#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include "lin_alg.h"
#include "input.h"


int main(){
    /*
     * Extracitn parameters
     */
    clock_t begin, end;
    begin = clock();
    double res = 1e-6;
    csc *A = malloc(sizeof(csc));
    dense_to_csc(m, n, As, A, 0);
    csc *P = malloc(sizeof(csc));
    dense_to_csc(n, n, Ps, P, 1);
    /*
     * Preconditoning 
     */
    double normP = max_abs(P);
    mat_normalize(P, normP);
    vec_normalize(q, n, normP);
    double *e = calloc(m, sizeof(double));
    inv_norms_t(A, e);
    premult_diag(A, e);
    entrywise_mult(b, e, m);  
    double normA = max_abs(A);
    mat_normalize(A, normA);
    vec_normalize(b, m, normA);
    double *Atb = calloc(n, sizeof(double));
    tr_mat_vec(A,b, Atb);
    /*
     * Forming KKT matrix
     */
    csc * top_half = malloc(sizeof(csc));
    add_identity(P, rho);
    hcat(P, transpose(A), top_half);
    csc * bottom_half = malloc(sizeof(csc));
    hcat(A, form_identity(m, -1.0 / rho), bottom_half);
    csc * kkt = malloc(sizeof(csc));
    vcat(top_half, bottom_half, kkt);
    ldl_matrix *ldl = malloc(sizeof(ldl_matrix));
    factorize(ldl, kkt);

    double *z = malloc((n+m)*sizeof(double));
    double *u1 = calloc(m, sizeof(double)); //returns a pointer to a malloced zero PREC
    double *u2 = calloc(n, sizeof(double));
    double *x = calloc(n+m, sizeof(double));
    double *Ax = malloc(n*sizeof(double));
    double *Atu1 = calloc(n, sizeof(double));
    double *rhs = calloc(n+m, sizeof(double));
    double *pre_proj = calloc(n, sizeof(double));
    double *residual = calloc(m, sizeof(double));
    double *x_work = malloc((n+m)*sizeof(double));
    double * x_best = calloc(n,sizeof(double));
    double * rr = calloc(n+m, sizeof(double));
    double * Pz = malloc(n*sizeof(double));
    double f_best = DBL_MAX;
    double f;
    res = res * normA * normA;
    end = clock();
    printf("TIME SPENT FOR PRECONDITIONING = %f\n", (double)(end- begin) / CLOCKS_PER_SEC);
    /* 
     * Iterations
     */
    begin = clock();
    for (int i=0; i<repeat; i++) {
        random_init(z, l1, l2, n); // Intialization
        if (i==0) {
          reset_u(z,n); // Always try zero as the first initialization
        }
        
        reset_u(u1, m);
        reset_u(u2, n);
        for (int iter=0; iter<max_iter; iter++) {
            // updating right hand side
            reset_u(Atu1, n);
            reset_u(rhs, n+m);
            tr_mat_vec(A, u1, Atu1);
            vec_add(rhs, z, n);
            vec_add(rhs, Atb, n);
            vec_sub(rhs, Atu1, n);
            vec_sub(rhs, u2, n);
            vec_normalize(rhs, n, 1.0/rho);
            vec_sub(rhs, q, n);
            backsolve(ldl, rhs, x, x_work);
            mat_vec(kkt, x, rr);
            vec_sub(rr, rhs, m+n);
            // update z
            reset_u(pre_proj, n);
            vec_add(pre_proj, x, n);
            vec_add(pre_proj, u2, n);
            projection(pre_proj, z, l1, l2, l3, n);
            // check if z is good:
            reset_u(residual, m);
            mat_vec(A, z, residual);
            vec_sub(residual, b, m);
            if (vec_norm(residual, m)< res) { 
                reset_u(Pz,n);
                mat_vec(P, z, Pz);
                f = normP * (.5 * (inner_prod(Pz, z, n) - rho * vec_norm(z,n)) + inner_prod(z, q, n) )+ r; 
                if (f < f_best) {
                    vec_copy(x_best, z, n);
                    f_best = f;
                  //  for (int i=0; i<n; i++)
                  //      printf("x[%d] = %f\n", i, x_best[i]);
                    printf("best f, repeat %d at iteration %d = %f\n", i, iter, f_best);
                }
            }
            // update u1, u2
            reset_u(Ax, m);
            mat_vec(A, x, Ax);
            for (int j=0; j<m; j++) {
                u1[j] += (Ax[j] - b[j]);
            }
            for (int j=0; j<n; j++) {
                u2[j] += (x[j] - z[j]);
            }
        }
    } 
    end = clock();
    printf("TIME SPENT FOR Iterations = %f\n", (double)(end- begin) / CLOCKS_PER_SEC);
    FILE *outf = fopen("output.txt", "w");
    for (int i=0;i<n;i++) {
        fprintf(outf, "%f, ", x_best[i]);
    }
    return(0);
}
