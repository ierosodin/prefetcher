CFLAGS = -msse2 --std gnu99 -O0 -Wall -Wextra

EXEC = naive_transpose see_transpose see_prefetch_transpose avx_transpose verify

GIT_HOOKS := .git/hooks/applied

$(GIT_HOOKS):
	@scripts/install-git-hooks
	@echo

all: $(GIT_HOOKS) $(EXEC)

SRCS_common = main.c

naive_transpose: $(SRCS_common)
	$(CC) $(CFLAGS) -o $@ \
	-D NAIVE \
	$(SRCS_common)

see_transpose: $(SRCS_common)
	$(CC) $(CFLAGS) -o $@ \
	-D SEE \
	$(SRCS_common)

see_prefetch_transpose: $(SRCS_common)
	$(CC) $(CFLAGS) -o $@ \
	-D SEE_PREFETCH \
	$(SRCS_common)

avx_transpose: $(SRCS_common)
	$(CC) $(CFLAGS) -mavx -o $@ \
	-D AVX \
	$(SRCS_common)

verify: verify.c
	$(CC) $(CFLAGS) -o $@ \
	-D AVX \
	verify.c

gencsv:
	for i in `seq 1604 4 1792`; do \
		printf "%d " $$i; \
		./see_transpose -w $$i -h $$i; \
		printf "\n";\
	done > result.csv
	gnuplot scripts/bench.gp

run: $(EXEC)

.PHONY: clean
clean:
