#include <stdio.h> // TODO: remove
#include <stdlib.h>
#include "lin_alg.h"
#include "SuiteSparse/AMD/Include/amd.h"
#include "SuiteSparse/LDL/Include/ldl.h"

double inner_prod(double *x, double *z, int n) {
    double r = 0;
    for (int i=0; i<n; i++) {
        r += (x[i] * z[i]);
    }
    return r;
}

void print_mat(csc *A, char *name){
    printf("n = %d, m = %d\n",A->n, A->m); 
    printf("%s = [\n", name);

    csc * At = transpose(A);

    int j,k;
    for (int i=0; i<A->n; i++){
        for(j=A->Ap[i]; j<A->Ap[i+1]; j++){     
            printf("Ai[%d] = %d, Ax[%d] = %f, Ap[%d] = %d\n", j, A->Ai[j], j, A->Ax[j], i, A->Ap[i]);
        }
    }
    printf("Ap[%d] = %d\n", A->n, A->Ap[A->n]);
  
    int count = 0;
    for (j=0; j<At->n; j++){
        for (k=0; k<At->m; k++) {
            if ((At->Ai[count] ==  k) && (count < At->Ap[j+1])) {
                printf("%f, ", At->Ax[count]);
                count++;
            }
            else {
               printf("%f, ", 0.);
            }
        }
        printf("\n");
    }
    printf("]\n\n");
}

void vec_copy(double *x, double *z, int n) {
    for (int i=0; i<n; i++) {
        x[i] = z[i];
    }
}

double vec_norm(double *z, int n) {
    double r = 0;
    for (int i=0; i<n; i++) {
        r += z[i]*z[i];
    }
    return r;
}

void vec_add(double *z1, double *z2, int n) {
    for (int i=0; i<n; i++) {
        z1[i] += z2[i];
    }
}

void vec_sub(double *z1, double *z2, int n) {
    for (int i=0; i<n; i++) {
        z1[i] -= z2[i];
    }
}

void reset_u(double *u, int n) {
    for (int i=0; i<n; i++) {
        u[i] = 0;
    }
}

void random_init(double *ans, int l1, int l2, int n) {
    for (int i=0; i<l1; i++) {
        ans[i] = (float) rand() / RAND_MAX;
    }
    for (int i=l1; i<l1+l2; i++) {
        ans[i] = 2 * ( (float) rand() / RAND_MAX ) - 1;
    }
    for (int i=l1+l2; i<n; i++) {
        ans[i] = (float) rand() / RAND_MAX;
    }
}

void projection(double *z, double *ans, int l1, int l2, int l3, int n) {
    for (int i=0; i<l1; i++) {
        if (z[i] > 0.5)
            ans[i] = 1;
        else
            ans[i] = 0;
    }
    for (int i=l1; i<l1+l2; i++) {
        if (z[i] > 0.5) {
            ans[i] = 1;
        }
        else if (z[i] < -.5) {
            ans[i] = -1;
        }
        else {
            ans[i] = 0;
        }
    }
    for (int i=l1+l2; i<l1+l2+l3; i++) {
        if (z[i] < 0) {
            ans[i] = 0;
        }
        else {
            ans[i] = z[i];
        }
    }
    for (int i=l1+l2+l3; i<n; i++) {
        ans[i] = z[i];
    }
}

void entrywise_mult(double *b, double *e, int n) {
    for (int i=0; i<n; i++) {
        b[i] *= e[i];
    }
}

// ez should be initialized zero
void inv_norms_t(csc *As, double *z) {
    for (int i=0; i<As->Ap[As->n]; i++) {
        if (As->Ax[i] > 0) {
            z[As->Ai[i]] += As->Ax[i];
        }
        if (As->Ax[i] < 0) {
            z[As->Ai[i]] -= As->Ax[i]; 
        }
    }
    for (int i=0; i<As->m; i++) {
        if (z[i] > EPS) {
            z[i] = 1/z[i];
        }
        else {
            z[i] = 1.0; //replace with max or similar
        }
    }
}

void vec_normalize(double * z, int l, double norm_factor) {
    for (int i=0; i<l; i++) {
        z[i] = z[i] / norm_factor;
    }
}

void mat_normalize(csc *As, double norm_factor) {
    for (int i=0; i<As->Ap[As->n]; i++) {
        As->Ax[i] = As->Ax[i] / norm_factor;
    }
}

double max_abs(csc *As) {
    double m = 0;
    for (int i=0; i<As->Ap[As->n]; i++) {
        if (As->Ax[i] > m) {
            m = As->Ax[i];
        }
        if (As->Ax[i] < -m) {
            m = - As->Ax[i];
        }
    }
    return m;
}

void dense_to_csc(int m, int n, double Ad[m][n], csc *As, int include_diags){
    int i, j, count = 0;
    As->n = n;
    As->m = m;
    As->Ap = malloc((n+1)*sizeof(int));

    As->Ap[0] = 0;
    for (j=0; j<n; j++){
        As->Ap[j+1] = As->Ap[j];
        for (i=0; i<m; i++){
            if ((Ad[i][j] > EPS || Ad[i][j] < -EPS) || (include_diags && i == j)){
                As->Ap[j+1]++;
            }
        }
    }

    As->Ai = malloc(As->Ap[n] * sizeof(int));
    As->Ax = malloc(As->Ap[n] * sizeof(double));

    for (j=0; j<n; j++){
        for (i=0; i<m; i++){
            if ((Ad[i][j] > EPS || Ad[i][j] < -EPS) || (include_diags && i == j)){
                As->Ai[count] = i;
                As->Ax[count] = Ad[i][j];
                count++;
            }
        }
    }
}

// Returns z' = x'*A'. (The vector z is output.)
void tr_mat_vec(csc *A, double *x, double *z){
    int j, k;
    for (j=0; j<A->n; j++){
        z[j] = 0;
        for (k=A->Ap[j]; k<A->Ap[j+1]; k++){
            z[j] += A->Ax[k] * x[A->Ai[k]];
        }
    }
}

// Returns z = A*x. (The vector z is output, and must be initialized to zeros.)
void mat_vec(csc *A, double *x, double *z){
    int j, k;
    for (j=0; j<A->n; j++){
        for (k=A->Ap[j]; k<A->Ap[j+1]; k++){
            z[A->Ai[k]] += A->Ax[k] * x[j];
        }
    }
}

void invert_p(int *p, int *pinv, int n){
    int i;
    for (i=0; i<n; i++){
        pinv[p[i]] = i;
    }
}

int compare_vec(double *x, double *z, int n){
    int i;
    for (i=0; i<n; i++){
        if (x[i] - z[i] > EPS || z[i] - x[i] > EPS){
            return(1);
        }
    }
    return(0);
}

void premult_diag(csc *A, double *f){
    int j, k;
    for (j=0; j<A->n; j++){
        for (k=A->Ap[j]; k<A->Ap[j+1]; k++){
            A->Ax[k] *= f[A->Ai[k]];
        }
    }
}

void postmult_diag(csc *A, double *e){
    int j, k;
    for (j=0; j<A->n; j++){
        for (k=A->Ap[j]; k<A->Ap[j+1]; k++){
            A->Ax[k] *= e[j];
        }
    }
}


void vcat(csc *A, csc *B, csc *Z){
    int j, k;
    int z_count=0;
    Z->n = A->n;
    Z->m = A->m + B->m;
    Z->Ap = malloc((A->n+1) * sizeof(int));
    Z->Ai = malloc((A->Ap[A->n] + B->Ap[B->n]) * sizeof(int));
    Z->Ax = malloc((A->Ap[A->n] + B->Ap[B->n]) * sizeof(double));

    for (j=1; j<A->n+1; j++){
        Z->Ap[j] = A->Ap[j] + B->Ap[j];
        for (k=A->Ap[j-1]; k<A->Ap[j]; k++){
            Z->Ai[z_count] = A->Ai[k];
            Z->Ax[z_count] = A->Ax[k];
            z_count++;
        }
        for (k=B->Ap[j-1]; k<B->Ap[j]; k++){
            Z->Ai[z_count] = B->Ai[k] + A->m;
            Z->Ax[z_count] = B->Ax[k];
            z_count++;
        }
    }
}

void hcat(csc *A, csc *B, csc *Z){
    int j, k;
    Z->n = A->n + B->n;
    Z->m = A->m;
    Z->Ap = malloc((A->n + B->n + 1) * sizeof(int));
    Z->Ai = malloc((A->Ap[A->n] + B->Ap[B->n]) * sizeof(int));
    Z->Ax = malloc((A->Ap[A->n] + B->Ap[B->n]) * sizeof(double));

    for (j=0; j<A->n; j++){
        Z->Ap[j] = A->Ap[j];
    }
    for (k=0; k<A->Ap[A->n]; k++){
        Z->Ai[k] = A->Ai[k];
        Z->Ax[k] = A->Ax[k];
    }
    for (j=0; j<B->n+1; j++){
        Z->Ap[j+A->n] = B->Ap[j] + A->Ap[A->n];
    }
    for (k=0; k<B->Ap[B->n]; k++){
        Z->Ai[k+A->Ap[A->n]] = B->Ai[k];
        Z->Ax[k+A->Ap[A->n]] = B->Ax[k];
    }
}

void amd(csc *A, int *p){
    amd_order(A->n, A->Ap, A->Ai, p, (double*)NULL, (double*)NULL);
}

void alloc_ldl(ldl_matrix *ldl, csc *A, int *p, int *pinv){
    ldl->n=A->n;
    ldl->Lp=calloc(A->n+1, sizeof(*ldl->Lp));
    ldl->Parent=calloc(A->n, sizeof(*ldl->Parent));
    ldl->Lnz=calloc(A->n, sizeof(*ldl->Lnz));
    ldl->Flag=calloc(A->n, sizeof(*ldl->Flag));
    ldl_symbolic(A->n, A->Ap, A->Ai, ldl->Lp, ldl->Parent, ldl->Lnz, ldl->Flag, p, pinv);
    ldl->Pattern=calloc(A->n, sizeof(*ldl->Pattern));
    ldl->Li=calloc(ldl->Lp[A->n], sizeof(*ldl->Li));
    ldl->Lx_factor=calloc(ldl->Lp[A->n], sizeof(*ldl->Lx_factor));
    ldl->D=calloc(A->n, sizeof(*ldl->D));
    ldl->Y=calloc(A->n, sizeof(*ldl->Y));
    ldl->Lx_solve=calloc(ldl->Lp[A->n], sizeof(*ldl->Lx_solve));
    ldl->D_inv=calloc(A->n, sizeof(*ldl->D_inv));
    ldl->p = p;
    ldl->pinv = pinv;
}

void factorize_numeric(csc *A, ldl_matrix *ldl){
    ldl_numeric(A->n, A->Ap, A->Ai, A->Ax, ldl->Lp, ldl->Parent, 
    ldl->Lnz, ldl->Li, ldl->Lx_factor, ldl->D, ldl->Y, ldl->Pattern, ldl->Flag, ldl->p, ldl->pinv);
    int i;
    for (i=0;i<ldl->n;i++){
        ldl->D_inv[i]=(double)(1/ldl->D[i]);
    }
    for (i=0;i<ldl->Lp[ldl->n];i++){
        ldl->Lx_solve[i]=(double) ldl->Lx_factor[i];
    }
}

void backsolve(ldl_matrix *ldl, double *b, double *x, double *x_work){
    ipermute(ldl->n, b, x_work, ldl->p);
    ldl_lsolve(ldl->n, x_work, ldl->Lp, ldl->Li, ldl->Lx_solve);
    int i;
    for(i=0;i<ldl->n;i++){
        x_work[i] = ldl->D_inv[i]*x_work[i];
    }
    ldl_ltsolve(ldl->n, x_work, ldl->Lp, ldl->Li, ldl->Lx_solve);
    permute(ldl->n, x_work, x, ldl->p);
}

void permute(int n, double *x, double *y, int *p){
    int i;
    for(i=0;i<n;i++){
        y[p[i]]=x[i];
    }
}

void ipermute(int n,double *x, double *y, int *p){
    int i;
    for(i=0;i<n;i++){
        y[i]=x[p[i]];
    }
}

void free_ldl(ldl_matrix *ldl){
    free(ldl->Lp);
    free(ldl->Parent);
    free(ldl->Lnz);
    free(ldl->Flag);
    free(ldl->Pattern);
    free(ldl->Li);
    free(ldl->D_inv);
    free(ldl->Y);
    free(ldl);
};

void free_csc(csc *A){
    free(A->Ap);
    free(A->Ai);
    free(A->Ax);
    free(A);
};


void factorize(ldl_matrix *ldl, csc *A){
    int *p = malloc(sizeof(int)*A->n);
    int *pinv = malloc(sizeof(int)*A->n);
    amd(A, p);
    invert_p(p, pinv, A->n);
    alloc_ldl(ldl, A, p, pinv);
    factorize_numeric(A, ldl);
}

void add_identity(csc *A, double rho){
    int j,k;
    for (j=0; j<A->n; j++){
        for (k=A->Ap[j]; k<A->Ap[j+1]; k++){
            if (A->Ai[k] == j){
                A->Ax[k] += rho;
            }
        }
    }
}

csc *form_identity(int n, double rho){
    int j;
    csc *A = malloc(sizeof(csc));
    A->n = n;
    A->Ap = malloc((n+1) * sizeof(int));
    A->Ai = malloc(n * sizeof(int));
    A->Ax = malloc(n * sizeof(double));
  
    for (j=0; j<A->n; j++){
        A->Ap[j] = j;
        A->Ai[j] = j;
        A->Ax[j] = rho;
    }
    A->Ap[n] = n;
    return(A);
}

csc *transpose(csc *A){
    int j, k, ind;
    csc *AT = malloc(sizeof(csc));
    AT->n = A->m;
    AT->m = A->n;
    AT->Ap = calloc(AT->n+1, sizeof(int));
    AT->Ai = malloc(A->Ap[A->n] * sizeof(int));
    AT->Ax = malloc(A->Ap[A->n] * sizeof(double));
    int *count = calloc(A->m, sizeof(int));

    for (k=0; k<A->Ap[A->n]; k++){
        AT->Ap[A->Ai[k]+1]++;
    }

    for (j=1; j<AT->n+1; j++){
        AT->Ap[j] += AT->Ap[j-1];
    }

    for (j=0; j<A->n; j++){
        for (k=A->Ap[j]; k<A->Ap[j+1]; k++){
            AT->Ai[ind = AT->Ap[A->Ai[k]] + count[A->Ai[k]]] = j;
            AT->Ax[ind] = A->Ax[k];
            count[A->Ai[k]]++;
        }
    }
    return(AT);
}


int compare_mat(csc *A, csc *B){
    int j, k;
    if (A->n != B->n){
        return 1;
    }
    for (j=0; j<A->n; j++){
        if (A->Ap[j] != B->Ap[j]){
            return 1;
        }
        for (k=A->Ap[j]; k<A->Ap[j+1]; k++){
            if (A->Ai[k] != B->Ai[k] || A->Ax[k] - B->Ax[k] > EPS || B->Ax[k] - A->Ax[k] < -EPS){
                return 1;
            }
        }
    }
    return(0);
}
