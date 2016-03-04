CC = gcc
CFLAGS = -g -Wall -O1 -funroll-loops
CFLAGS += -I ../SuiteSparse/include/
LDFLAGS = -L ../SuiteSparse/AMD/Lib/ -L ../SuiteSparse/SuiteSparse_config/ -lamd -lm -lsuitesparseconfig

default: amd_test

debug: CFLAGS += -g

amd_test: amd_test.c
	$(CC) amd_test.c $(CFLAGS) $(LDFLAGS) -o amd_test

LDL/Lib/libldl.a:
	cd LDL && make

# In make's default rules, a .o automatically depends on its .c file
# (so editing the .c will cause recompilation into its .o file).
# The line below creates additional dependencies, most notably that it
# will cause the .c to reocmpiled if any included .h file changes.

# Phony means not a "real" target, it doesn't build anything
# The phony target "clean" that is used to remove all compiled object files.

.PHONY: clean
.PHONY: cleanall
.PHONY: re

cleanall:
	@rm -rf $(TARGETS) $(OBJECTS) core Makefile.dependencies *.o *.a *.h.gch
	cd LDL && make purge
	cd SuiteSparse_config && make purge

clean:
	@rm -rf $(TARGETS) $(OBJECTS) core Makefile.dependencies *.o *.a *.h.gch

re: clean amd_test
