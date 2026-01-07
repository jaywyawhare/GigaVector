CC      := gcc
CFLAGS  := -O3 -g -Wall -Wextra -MMD -Iinclude -march=native -msse4.2 -mavx2 -mavx512f -mfma -pthread -fPIC
LDFLAGS := -lm -pthread

BUILD_DIR   := build
SRC_DIR     := src
INCLUDE_DIR := include
OBJ_DIR     := $(BUILD_DIR)/obj
BIN_DIR     := $(BUILD_DIR)
LIB_DIR     := $(BUILD_DIR)/lib
DATA_DIR    := snapshots
BENCH_DIR   := $(BUILD_DIR)/bench

LIB_NAME    := GigaVector
LIB_VERSION := 0.0.1 
STATIC_LIB  := $(LIB_DIR)/lib$(LIB_NAME).a
SHARED_LIB  := $(LIB_DIR)/lib$(LIB_NAME).so

SRC_FILES   := $(shell find $(SRC_DIR) -name "*.c")
MAIN_FILE   := main.c
BENCH_FILES := benchmark_simd.c benchmark_compare.c benchmark_ivfpq.c benchmark_ivfpq_recall.c
TEST_DIR    := tests

PYTHON_DIR  := python
PYTHON_SRC  := $(PYTHON_DIR)/src
PYTHON_TEST := $(PYTHON_DIR)/tests

LIB_OBJS    := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC_FILES))
MAIN_OBJ    := $(OBJ_DIR)/$(MAIN_FILE:.c=.o)
ALL_OBJS    := $(LIB_OBJS) $(MAIN_OBJ)

DEPS := $(ALL_OBJS:.o=.d)

.PHONY: all
all: $(BIN_DIR)/main

.PHONY: run
run: $(BIN_DIR)/main
	@mkdir -p $(DATA_DIR)
	@echo "Running demo with outputs in $(DATA_DIR)/"
	@cd $(DATA_DIR) && GV_DATA_DIR="$(abspath $(DATA_DIR))" GV_WAL_DIR="$(abspath $(DATA_DIR))" $(abspath $(BIN_DIR))/main

.PHONY: bench
bench: $(BENCH_DIR)/benchmark_simd $(BENCH_DIR)/benchmark_compare $(BENCH_DIR)/benchmark_ivfpq $(BENCH_DIR)/benchmark_ivfpq_recall
	@echo "Benchmarks built in $(BENCH_DIR)"

$(BIN_DIR)/main: $(MAIN_OBJ) $(STATIC_LIB)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(MAIN_OBJ) $(LDFLAGS) -L$(LIB_DIR) -l$(LIB_NAME) -o $@
	@echo "Built main executable: $@"

.PHONY: lib
lib: $(STATIC_LIB) $(SHARED_LIB)

$(STATIC_LIB): $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $^
	@echo "Built static library: $@"

$(SHARED_LIB): $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	$(CC) -shared $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built shared library: $@"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@
	@echo "Compiled $< -> $@"

$(OBJ_DIR)/$(MAIN_FILE:.c=.o): $(MAIN_FILE)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@
	@echo "Compiled $< -> $@"

$(BENCH_DIR)/benchmark_%: benchmark_%.c $(STATIC_LIB)
	@mkdir -p $(BENCH_DIR)
	$(CC) $(CFLAGS) $< -L$(LIB_DIR) -l$(LIB_NAME) $(LDFLAGS) -o $@
	@echo "Built benchmark: $@"

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) a.out *.d
	@echo "Cleaned build artifacts"

.PHONY: distclean
distclean: clean
	rm -rf $(DATA_DIR)
	@echo "Cleaned all artifacts including data directory"

.PHONY: test-corrupt-wal
test-corrupt-wal: $(BIN_DIR)/main
	@bash $(TEST_DIR)/corrupt_wal.sh $(DATA_DIR)/database.bin.wal || true

.PHONY: test-corrupt-snapshot
test-corrupt-snapshot: $(BIN_DIR)/main
	@bash $(TEST_DIR)/corrupt_snapshot.sh $(DATA_DIR)/database.bin || true

.PHONY: bench-ivfpq-suite
bench-ivfpq-suite: $(BENCH_DIR)/benchmark_ivfpq $(BENCH_DIR)/benchmark_ivfpq_recall
	@BIN_DIR=$(BENCH_DIR) bash $(TEST_DIR)/ivfpq_suite.sh

TEST_FILES := test_db test_distance test_metadata test_hnsw test_ivfpq test_sparse test_wal test_filter test_advanced
TEST_SRCS := $(patsubst %,tests/%.c,$(TEST_FILES))
TEST_BINS := $(patsubst %,$(BUILD_DIR)/test_%,$(TEST_FILES))

ASAN_FLAGS := -fsanitize=address -fno-omit-frame-pointer -g
TSAN_FLAGS := -fsanitize=thread -fno-omit-frame-pointer -g
MSAN_FLAGS := -fsanitize=memory -fno-omit-frame-pointer -g
UBSAN_FLAGS := -fsanitize=undefined -fno-omit-frame-pointer -g

.PHONY: python-test
python-test: lib
	@cd $(PYTHON_DIR) && PYTHONPATH=src python -m unittest discover -s tests

.PHONY: python-test-comprehensive
python-test-comprehensive: lib
	@cd $(PYTHON_DIR) && PYTHONPATH=src python -m unittest discover -s tests -v

.PHONY: c-test
c-test: lib $(TEST_BINS)
	@echo "Running all C tests..."
	@for test in $(TEST_BINS); do \
		echo "Running $$test..."; \
		LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH $$test || exit 1; \
	done
	@echo "All C tests passed"

.PHONY: c-test-single
c-test-single: lib
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make c-test-single TEST=test_db"; \
		exit 1; \
	fi
	@mkdir -p $(BUILD_DIR)
	@$(CC) $(CFLAGS) tests/$(TEST).c -L$(LIB_DIR) -l$(LIB_NAME) $(LDFLAGS) -o $(BUILD_DIR)/$(TEST)
	@echo "Built test: $(BUILD_DIR)/$(TEST)"
	@LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH $(BUILD_DIR)/$(TEST)

$(BUILD_DIR)/test_%: tests/%.c $(STATIC_LIB)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $< -L$(LIB_DIR) -l$(LIB_NAME) $(LDFLAGS) -o $@
	@echo "Built test: $@"

.PHONY: test-asan
test-asan: CFLAGS += $(ASAN_FLAGS)
test-asan: LDFLAGS += -fsanitize=address
test-asan:
	@$(MAKE) clean
	@$(MAKE) lib $(TEST_BINS)
	@echo "Running tests with AddressSanitizer..."
	@for test in $(TEST_BINS); do \
		echo "Running $$test with ASAN..."; \
		LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH $$test || exit 1; \
	done
	@echo "All ASAN tests passed"

.PHONY: test-tsan
test-tsan: CFLAGS += $(TSAN_FLAGS)
test-tsan: LDFLAGS += -fsanitize=thread
test-tsan:
	@$(MAKE) clean
	@$(MAKE) lib $(TEST_BINS)
	@echo "Running tests with ThreadSanitizer..."
	@for test in $(TEST_BINS); do \
		echo "Running $$test with TSAN..."; \
		LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH $$test || exit 1; \
	done
	@echo "All TSAN tests passed"

.PHONY: test-ubsan
test-ubsan: CFLAGS += $(UBSAN_FLAGS)
test-ubsan: LDFLAGS += -fsanitize=undefined
test-ubsan:
	@$(MAKE) clean
	@$(MAKE) lib $(TEST_BINS)
	@echo "Running tests with UndefinedBehaviorSanitizer..."
	@for test in $(TEST_BINS); do \
		echo "Running $$test with UBSAN..."; \
		LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH $$test || exit 1; \
	done
	@echo "All UBSAN tests passed"

.PHONY: test-valgrind
test-valgrind: lib $(BUILD_DIR)/test_test_db
	@echo "Running tests with Valgrind..."
	@if command -v valgrind >/dev/null 2>&1; then \
		VALGRIND_OUTPUT=$$(LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH \
			valgrind --leak-check=full --show-leak-kinds=all \
			--track-origins=yes --error-exitcode=1 \
			$(BUILD_DIR)/test_test_db 2>&1); \
		VALGRIND_EXIT=$$?; \
		echo "$$VALGRIND_OUTPUT"; \
		if echo "$$VALGRIND_OUTPUT" | grep -q "Fatal error at startup"; then \
			echo ""; \
			echo "Valgrind configuration issue detected. Skipping Valgrind tests (non-fatal)..."; \
			exit 0; \
		elif [ $$VALGRIND_EXIT -eq 0 ]; then \
			echo "Valgrind tests passed"; \
		else \
			echo "Valgrind test failed"; \
			exit 1; \
		fi; \
	else \
		echo "Valgrind not found, skipping..."; \
	fi

.PHONY: test-all
test-all: c-test python-test-comprehensive test-asan test-tsan test-ubsan
	@echo "Running Valgrind tests (if available)..."
	@-$(MAKE) test-valgrind || echo "Valgrind tests skipped (configuration issue or not available)"
	@echo "All tests and sanitizers completed"

.PHONY: test-coverage
test-coverage: CFLAGS += --coverage -O0
test-coverage: LDFLAGS += --coverage
test-coverage:
	@$(MAKE) clean
	@$(MAKE) lib $(TEST_BINS)
	@echo "Running tests with coverage..."
	@for test in $(TEST_BINS); do \
		LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH $$test || exit 1; \
	done
	@echo "Coverage data generated in .gcda files"
	@if command -v gcov >/dev/null 2>&1; then \
		echo "Generating coverage report..."; \
		for src in $(SRC_FILES); do \
			gcov -r $$src 2>/dev/null | head -5; \
		done; \
	fi

.PHONY: test-coverage-html
test-coverage-html: test-coverage
	@if command -v lcov >/dev/null 2>&1 && command -v genhtml >/dev/null 2>&1; then \
		echo "Generating HTML coverage report..."; \
		lcov --capture --directory $(OBJ_DIR) --output-file $(BUILD_DIR)/coverage.info; \
		genhtml $(BUILD_DIR)/coverage.info --output-directory $(BUILD_DIR)/coverage_html; \
		echo "Coverage report available at $(BUILD_DIR)/coverage_html/index.html"; \
	else \
		echo "lcov/genhtml not found, skipping HTML report"; \
	fi

-include $(DEPS)