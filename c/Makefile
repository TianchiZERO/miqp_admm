CC = gcc
CFLAGS = -Wall -O1 -funroll-loops
CFLAGS += -I SuiteSparse/include/ -std=gnu99 
LDFLAGS = -L SuiteSparse/LDL/Lib/ -L SuiteSparse/AMD/Lib -L SuiteSparse/SuiteSparse_config -lldl -lamd -lm -lsuitesparseconfig

default: lin_alg.o
	$(CC)  lin_alg.o main.c $(CFLAGS) $(LDFLAGS) -o main

debug: CFLAGS += -g

test: tests/lin_alg_tests.c lin_alg.h lin_alg.o
	$(CC) lin_alg.h lin_alg.o tests/lin_alg_tests.c $(CFLAGS) $(LDFLAGS) -o test

lin_alg.o: SuiteSparse/LDL/Lib/libldl.a SuiteSparse/AMD/Lib/libamd.a
	$(CC) lin_alg.h $(CFLAGS) $(LDFLAGS) -c lin_alg.c

SuiteSparse/LDL/Lib/libldl.a:
	cd SuiteSparse/LDL && make

SuiteSparse/AMD/Lib/libldl.a:
	cd SuiteSparse/AMD && make


.PHONY: clean
.PHONY: cleanall
.PHONY: re
.PHONY: all

cleanall: clean
	cd SuiteSparse/ && make purge

clean:
	@rm -rf $(TARGETS) $(OBJECTS) core Makefile.dependencies *.o *.a *.h.gch test

re: clean all

all: default test
