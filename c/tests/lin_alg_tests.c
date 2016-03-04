#include <stdio.h>
#include <stdlib.h>
#include "../lin_alg.h"
#include "matrices.c"


int test_max_abs(){
  csc *As = malloc(sizeof(csc));
  dense_to_csc(m, n, A, As, 0);
  return(abs(max_abs(As) - 2.111017) > EPS);
}


int test_mat_vec(){
  double *z = calloc(m, sizeof(double));
  csc *As = malloc(sizeof(csc));
  dense_to_csc(m, n, A, As, 0);

  mat_vec(As, x, z);
  return(compare_vec(Ax, z, m));
}

int test_tr_mat_vec(){
  double *z = calloc(n, sizeof(double));
  csc *As = malloc(sizeof(csc));
  dense_to_csc(m, n, A, As, 0);

  tr_mat_vec(As, y, z);
  return(compare_vec(ATy, z, n));
}

int test_premult_diag(){
  csc *As = malloc(sizeof(csc));
  csc *FAs = malloc(sizeof(csc));
  dense_to_csc(m, n, A, As, 0);
  dense_to_csc(m, n, FA, FAs, 0);

  premult_diag(As, y);
  return(compare_mat(As, FAs));
}

int test_postmult_diag(){
  csc *As = malloc(sizeof(csc));
  csc *AEs = malloc(sizeof(csc));
  dense_to_csc(m, n, A, As, 0);
  dense_to_csc(m, n, AE, AEs, 0);

  postmult_diag(As, x);
  return(compare_mat(As, AEs));
}

int test_vcat(){
  csc *As = malloc(sizeof(csc));
  csc *Bs = malloc(sizeof(csc));
  csc *Z = malloc(sizeof(csc));
  csc *ABs = malloc(sizeof(csc));
  dense_to_csc(m, n, A, As, 0);
  dense_to_csc(m, n, B, Bs, 0);
  dense_to_csc(2*m, n, AvcatB, ABs, 0);

  vcat(As, Bs, Z);
  return(compare_mat(ABs, Z));
}

int test_hcat(){
  csc *As = malloc(sizeof(csc));
  csc *Bs = malloc(sizeof(csc));
  csc *Z = malloc(sizeof(csc));
  csc *ABs = malloc(sizeof(csc));
  dense_to_csc(m, n, A, As, 0);
  dense_to_csc(m, n, B, Bs, 0);
  dense_to_csc(m, 2*n, AhcatB, ABs, 0);

  hcat(As, Bs, Z);
  return(compare_mat(ABs, Z));
}

int test_ldl(){
  csc *Ps = malloc(sizeof(csc));
  dense_to_csc(n, n, P, Ps, 0);
  ldl_matrix *ldl = malloc(sizeof(ldl_matrix));
  double *z = malloc(n * sizeof(double));
  double *x_work = malloc(n * sizeof(double));

  // MAKE A NULL PERMUTATION:
  int *p = malloc(n * sizeof(int));
  int *pinv = malloc(n * sizeof(int));
  int i;
  for(i=0; i<n; i++){
    p[i] = i;
  }
  invert_p(p, pinv, n);

  alloc_ldl(ldl, Ps, p, pinv);
  factorize_numeric(Ps, ldl);

  backsolve(ldl, x, z, x_work);
  return(compare_vec(Pinvx, z, n));
}

void test_amd(){
  csc *Ps = malloc(sizeof(csc));
  ldl_matrix *ldl_orig = malloc(sizeof(ldl_matrix));
  ldl_matrix *ldl_perm = malloc(sizeof(ldl_matrix));
  dense_to_csc(n, n, P, Ps, 0);

  // MAKE A NULL PERMUTATION:
  int *p = malloc(n * sizeof(int));
  int *pinv = malloc(n * sizeof(int));
  int i;

  for(i=0; i<n; i++){
    p[i] = i;
  }
  invert_p(p, pinv, n);

  alloc_ldl(ldl_orig, Ps, p, pinv);
  factorize_numeric(Ps, ldl_orig);
  printf("\n  nnz in lower half of L factor, without permutation: %d\n", ldl_orig->Lp[ldl_orig->n]);

  amd(Ps, p);
  invert_p(p, pinv, n);

  alloc_ldl(ldl_perm, Ps, p, pinv);
  factorize_numeric(Ps, ldl_perm);
  printf(  "  nnz in lower half of L factor,    with permutation: %d\n\n", ldl_perm->Lp[ldl_perm->n]);
}


int test_factorize(){
  csc *Ps = malloc(sizeof(csc));
  dense_to_csc(n, n, P, Ps, 0);
  ldl_matrix *ldl = malloc(sizeof(ldl_matrix));
  double *z = malloc(n * sizeof(double));
  double *x_work = malloc(n * sizeof(double));

  factorize(ldl, Ps);
  backsolve(ldl, x, z, x_work);

  double *rr = calloc(n, sizeof(double)); 
  mat_vec(Ps, z, rr);
  vec_sub(rr, x, n);
  printf("residual norm = %f\n", vec_norm(rr,n));

  return(compare_vec(Pinvx, z, n));
}

int test_add_identity(){
  csc *Ps = malloc(sizeof(csc));
  csc *PrhoIs = malloc(sizeof(csc));
  dense_to_csc(n, n, P, Ps, 1);
  dense_to_csc(n, n, PrhoI, PrhoIs, 0);
  add_identity(Ps, rho);
  return(compare_mat(Ps, PrhoIs));
}

int test_form_identity(){
  csc *rhoIs = malloc(sizeof(csc));
  dense_to_csc(n, n, rhoI, rhoIs, 0);

  csc *As = form_identity(n, rho);
  return(compare_mat(As, rhoIs));
}


int test_transpose(){
  csc *As = malloc(sizeof(csc));
  dense_to_csc(m, n, A, As, 0);
  csc *ATs = malloc(sizeof(csc));
  dense_to_csc(n, m, AT, ATs, 0);
  print_mat(As,"A");
  print_mat(ATs, "At_real");

  csc *ATs_test = transpose(As);
  print_mat(ATs_test, "At_found");
  return(compare_mat(ATs, ATs_test));
}



void amd(csc *P, int *p);

int main(){

  printf("\nA value of 1 indicates failure.  0 indicates success.\n\n");
  printf("                   maximum absolute value result: %d\n", test_max_abs());
  printf("             matrix-vector multiplication result: %d\n", test_mat_vec());
  printf("  transposed matrix-vector multiplication result: %d\n", test_tr_mat_vec());
  printf("     premultiplication by diagonal matrix result: %d\n", test_premult_diag());
  printf("    postmultiplication by diagonal matrix result: %d\n", test_postmult_diag());
  printf("                   vertical concatenation result: %d\n", test_vcat());
  printf("                 horizontal concatenation result: %d\n", test_hcat());
  printf("                                LDL solve result: %d\n", test_ldl());
  printf("        LDL solve result, using wrapper function: %d\n", test_factorize());
  printf("                             add identity matrix: %d\n", test_add_identity());
  printf("                            form identity matrix: %d\n", test_form_identity());
  printf("                                transpose result: %d\n", test_transpose());
  test_amd();

  return(0);

}
