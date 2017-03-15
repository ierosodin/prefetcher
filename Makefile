CFLAGS = -msse2 --std gnu99 -O0 -Wall -Wextra

EXEC = naive_transpose see_transpose see_prefetch_transpose

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

run: $(EXEC)

.PHONY: clean
clean:
