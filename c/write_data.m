function write_data(file, rho, max_iter, repeat)
eval(file)
[m,n] = size(A);
name = 'input.h';
delete(name);
fi = fopen(name,'w');
fprintf(fi,'int m = %u;\n', m);
fprintf(fi,'int n = %u;\n', n);
fprintf(fi,'int l1 = %u;\n', l1);
fprintf(fi,'int l2 = %u;\n', l2);
fprintf(fi,'int l3 = %u;\n', l3);
fprintf(fi,'double rho = %f;\n', rho);
fprintf(fi,'int max_iter = %d;\n', max_iter);
fprintf(fi,'int repeat = %d;\n', repeat);
fprintf(fi,'\n');
fprintf(fi,'double r = %f;\n', r);
write_vec(fi, q, 'q');
write_vec(fi, b, 'b');
write_mat(fi, P, 'Ps');
write_mat(fi, A, 'As');
end

function write_vec(fi, x, name) 
  n = length(x);
  fprintf(fi, 'double %s[%d] = {', name, n);
  for i=1:n
    fprintf(fi, '%f, ',x(i));
  end
  fprintf(fi, '};\n\n');
end

function write_mat(fi, A, name)
  [m, n] = size(A);
  fprintf(fi, 'double %s[%d][%d] = {\n', name, m, n);
  for i=1:m
      fprintf(fi, '  {');
      for j=1:n 
          fprintf(fi, '%f, ', A(i,j));
      end
      fprintf(fi, '},\n');
  end
  fprintf(fi, '};\n\n');
end
