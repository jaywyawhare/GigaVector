CC      := gcc
BASE_CFLAGS := -O3 -g -Wall -Wextra -MMD -Iinclude -pthread -fPIC
SIMD_FLAGS ?=
HARDENING_FLAGS ?=
CURL_FLAGS ?=
OPENSSL_FLAGS ?=
ONNX_FLAGS ?=
CFLAGS  := $(BASE_CFLAGS) $(SIMD_FLAGS) $(HARDENING_FLAGS) $(CURL_FLAGS) $(OPENSSL_FLAGS) $(ONNX_FLAGS)
LDFLAGS := -lm -pthread $(if $(CURL_FLAGS),-lcurl,) $(if $(OPENSSL_FLAGS),-lssl -lcrypto,)

BUILD_DIR   := build
SRC_DIR     := src
INCLUDE_DIR := include
OBJ_DIR     := $(BUILD_DIR)/obj
BIN_DIR     := $(BUILD_DIR)
LIB_DIR     := $(BUILD_DIR)/lib
DATA_DIR    := snapshots
BENCH_DIR   := $(BUILD_DIR)/bench

LIB_NAME    := GigaVector
LIB_VERSION := 0.8.25
STATIC_LIB  := $(LIB_DIR)/lib$(LIB_NAME).a

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SHARED_LIB       := $(LIB_DIR)/lib$(LIB_NAME).dylib
  SHARED_LIB_FLAGS := -dynamiclib
else
  SHARED_LIB       := $(LIB_DIR)/lib$(LIB_NAME).so
  SHARED_LIB_FLAGS := -shared
endif

SRC_FILES   := $(shell find $(SRC_DIR) -name "*.c")
MAIN_FILE   := main.c
BENCH_FILES := benchmarks/benchmark_simd.c benchmarks/benchmark_compare.c benchmarks/benchmark_ivfpq.c benchmarks/benchmark_ivfpq_recall.c benchmarks/bench_ivfdisk.c
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
bench: $(BENCH_DIR)/benchmark_simd $(BENCH_DIR)/benchmark_compare $(BENCH_DIR)/benchmark_ivfpq $(BENCH_DIR)/benchmark_ivfpq_recall $(BENCH_DIR)/bench_ivfdisk
	@echo "Benchmarks built in $(BENCH_DIR)"

.PHONY: bench-ivfdisk
bench-ivfdisk: $(BENCH_DIR)/bench_ivfdisk
	@echo "=== IVFDisk smoke benchmark (10k x 128) ==="
	@LD_LIBRARY_PATH=$(LIB_DIR) $(BENCH_DIR)/bench_ivfdisk 10000 128 64 32 30 1

.PHONY: bench-ivfdisk-full
bench-ivfdisk-full: $(BENCH_DIR)/bench_ivfdisk
	@echo "=== IVFDisk full benchmark (1M x 128) — expect long runtime ==="
	@LD_LIBRARY_PATH=$(LIB_DIR) $(BENCH_DIR)/bench_ivfdisk 1000000 128 1024 64 100 0

$(BIN_DIR)/main: $(MAIN_OBJ) $(STATIC_LIB)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(MAIN_OBJ) $(STATIC_LIB) $(LDFLAGS) -o $@
	@echo "Built main executable: $@"

.PHONY: lib
lib: $(STATIC_LIB) $(SHARED_LIB)

$(STATIC_LIB): $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $^
	@echo "Built static library: $@"

$(SHARED_LIB): $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	$(CC) $(SHARED_LIB_FLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built shared library: $@"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@
	@echo "Compiled $< -> $@"

$(OBJ_DIR)/$(MAIN_FILE:.c=.o): $(MAIN_FILE)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@
	@echo "Compiled $< -> $@"

$(BENCH_DIR)/benchmark_%: benchmarks/benchmark_%.c $(STATIC_LIB)
	@mkdir -p $(BENCH_DIR)
	$(CC) $(CFLAGS) $< -L$(LIB_DIR) -l$(LIB_NAME) $(LDFLAGS) -o $@
	@echo "Built benchmark: $@"

$(BENCH_DIR)/bench_ivfdisk: benchmarks/bench_ivfdisk.c $(STATIC_LIB)
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

# Corrupt-input resilience: feed corrupted WAL + snapshot bytes to the loaders
# and assert they return a defined error (no crash). Real exit code — a crash or
# silent-accept fails the build (unlike the descriptive scripts above).
.PHONY: test-corrupt-resilience
test-corrupt-resilience: lib $(BUILD_DIR)/storage/test_corrupt_resilience$(EXE_EXT)
	@echo "Running corrupt-WAL / corrupt-snapshot resilience check..."
	@LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH $(BUILD_DIR)/storage/test_corrupt_resilience

.PHONY: bench-ivfpq-suite
bench-ivfpq-suite: $(BENCH_DIR)/benchmark_ivfpq $(BENCH_DIR)/benchmark_ivfpq_recall
	@BIN_DIR=$(BENCH_DIR) bash $(TEST_DIR)/ivfpq_suite.sh

TEST_SRCS := $(shell find $(TEST_DIR) -name "test_*.c")
ifeq ($(OS),Windows_NT)
EXE_EXT := .exe
else ifneq (,$(findstring mingw,$(CC)))
EXE_EXT := .exe
else
EXE_EXT :=
endif
TEST_BINS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%$(EXE_EXT),$(TEST_SRCS))

# Deterministic simulation tests (DST). Globbed so new tests/dst/*.c can't drift
# out of CI: `make dst-test` builds and runs every one.
DST_SRCS := $(sort $(wildcard $(TEST_DIR)/dst/test_*.c))
DST_BINS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%$(EXE_EXT),$(DST_SRCS))

.PHONY: dst-test
dst-test: lib $(DST_BINS)
	@echo "Running DST oracles..."
	@for test in $(DST_BINS); do \
		echo "Running $$test..."; \
		LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH $$test || exit 1; \
	done
	@echo "All DST oracles passed"

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
		echo "Usage: make c-test-single TEST=storage/test_db"; \
		exit 1; \
	fi
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/$$(dirname "$(TEST)")
	@$(CC) $(CFLAGS) $(TEST_DIR)/$(TEST).c -L$(LIB_DIR) -l$(LIB_NAME) $(LDFLAGS) -Wl,-rpath,$(abspath $(LIB_DIR)) -o $(BUILD_DIR)/$(TEST)
	@echo "Built test: $(BUILD_DIR)/$(TEST)"
	@$(BUILD_DIR)/$(TEST)

$(BUILD_DIR)/%$(EXE_EXT): $(TEST_DIR)/%.c $(STATIC_LIB)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $< -L$(LIB_DIR) -l$(LIB_NAME) $(LDFLAGS) -Wl,-rpath,$(abspath $(LIB_DIR)) -o $@
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
test-valgrind: lib $(BUILD_DIR)/storage/test_db
	@echo "Running tests with Valgrind..."
	@if command -v valgrind >/dev/null 2>&1; then \
		VALGRIND_OUTPUT=$$(LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH \
			valgrind --leak-check=full --show-leak-kinds=all \
			--track-origins=yes --error-exitcode=1 \
			$(BUILD_DIR)/storage/test_db 2>&1); \
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

# Curated valgrind run over memory-sensitive binaries. Unlike `test-valgrind`
# (which runs a single binary and whose failure is swallowed by `test-all`),
# this uses --error-exitcode=1 --leak-check=full so any leak/error fails the
# build, and it is wired into CI as its own step. Kept to a reasonable set to
# bound runtime.
# Curated to binaries that are valgrind-clean today so the gate is meaningful
# and green: any NEW leak fails CI.
#   test_flat / test_corrupt_resilience are now clean and gated:
#     - test_flat: tests free their owned search results (gv_search_results_free).
#     - test_corrupt_resilience: the db_open_from_memory corrupt/truncated-input
#       error paths now fully tear down the partially-built db.
#   test_hnsw / test_db are NOT yet gated: they still hit a pre-existing LIBRARY
#   leak in db_add_vector (database.c) — for the HNSW/IVFPQ/IVFFLAT/IVFSQ8/
#   IVFTURBOQUANT index families, db_add_vector allocates a GV_Vector shell whose
#   data is copied into soa_storage by *_insert(), but (unlike flat/pq/lsh/rabitq
#   inserts) the shell is never freed on success. Add them here once that library
#   ownership bug is fixed.
VALGRIND_CORE_TESTS := \
	index/test_flat \
	storage/test_corrupt_resilience \
	index/test_ivfpq \
	features/test_recommend \
	search/test_group_search \
	search/test_mmr \
	core/test_alloc_fail \
	index/test_ivfdisk \
	index/test_exact_search \
	index/test_ivfflat \
	index/test_lsh \
	index/test_pq \
	index/test_rabitq \
	search/test_hybrid_search \
	search/test_score_threshold \
	search/test_filter \
	storage/test_sparse \
	storage/test_advanced \
	storage/test_memory \
	storage/test_memory_consolidation \
	storage/test_memory_links
VALGRIND_CORE_BINS := $(patsubst %,$(BUILD_DIR)/%$(EXE_EXT),$(VALGRIND_CORE_TESTS))

.PHONY: test-valgrind-core
test-valgrind-core: lib $(VALGRIND_CORE_BINS)
	@echo "Running curated valgrind memory checks..."
	@if ! command -v valgrind >/dev/null 2>&1; then \
		echo "Valgrind not found, skipping..."; exit 0; \
	fi
	@for test in $(VALGRIND_CORE_BINS); do \
		echo "==> valgrind $$test"; \
		LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH \
			valgrind --leak-check=full --error-exitcode=1 \
			--errors-for-leak-kinds=definite,indirect \
			$$test || exit 1; \
		rm -rf $${TMPDIR:-/tmp}/gv_* 2>/dev/null || true; \
	done
	@echo "All curated valgrind checks passed"

.PHONY: test-all
test-all: c-test python-test-comprehensive test-asan test-tsan test-ubsan
	@echo "Running Valgrind tests (if available)..."
	@-$(MAKE) test-valgrind || echo "Valgrind tests skipped (configuration issue or not available)"
	@echo "All tests and sanitizers completed"

.PHONY: test-coverage
test-coverage:
	@$(MAKE) clean
	@echo "Building with coverage instrumentation (using -O0 for accurate coverage)..."
	@$(MAKE) CFLAGS="-O0 -g -Wall -Wextra -MMD -Iinclude -pthread -fPIC -DHAVE_CURL --coverage" \
		LDFLAGS="-lm -pthread -lcurl --coverage" lib $(TEST_BINS)
	@echo "Running tests with coverage..."
	@for test in $(TEST_BINS); do \
		LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH $$test || exit 1; \
	done
	@echo "Coverage data generated in .gcda files in $(OBJ_DIR)/"
	@if command -v lcov >/dev/null 2>&1; then \
		echo ""; \
		echo "Generating coverage summary..."; \
		lcov --capture --directory $(OBJ_DIR) --output-file $(BUILD_DIR)/coverage.info --quiet 2>/dev/null; \
		lcov --remove $(BUILD_DIR)/coverage.info '/usr/*' '*/tests/*' --output-file $(BUILD_DIR)/coverage.info --quiet 2>/dev/null; \
		echo ""; \
		lcov --summary $(BUILD_DIR)/coverage.info 2>/dev/null | tail -4; \
		echo ""; \
		echo "For detailed HTML report, run: make test-coverage-html"; \
	else \
		echo ""; \
		echo "For detailed coverage reports, install 'lcov':"; \
		echo "  Ubuntu/Debian: sudo apt-get install lcov"; \
		echo "  Fedora/RHEL:   sudo dnf install lcov"; \
		echo "  macOS:         brew install lcov"; \
		echo "  Arch Linux:    sudo pacman -S lcov"; \
		echo ""; \
		echo "Then run: make test-coverage-html"; \
	fi

.PHONY: test-coverage-html
test-coverage-html: test-coverage
	@if command -v lcov >/dev/null 2>&1 && command -v genhtml >/dev/null 2>&1; then \
		echo "Generating HTML coverage report..."; \
		lcov --capture --directory $(OBJ_DIR) --output-file $(BUILD_DIR)/coverage.info --quiet; \
		lcov --remove $(BUILD_DIR)/coverage.info '/usr/*' '*/tests/*' --output-file $(BUILD_DIR)/coverage.info --quiet; \
		genhtml $(BUILD_DIR)/coverage.info --output-directory $(BUILD_DIR)/coverage_html --quiet; \
		echo ""; \
		echo "Coverage report available at: $(BUILD_DIR)/coverage_html/index.html"; \
		echo "Open with: xdg-open $(BUILD_DIR)/coverage_html/index.html  (Linux)"; \
		echo "           open $(BUILD_DIR)/coverage_html/index.html     (macOS)"; \
	else \
		echo ""; \
		echo "ERROR: lcov/genhtml not found. Install to generate HTML coverage reports:"; \
		echo "  Ubuntu/Debian: sudo apt-get install lcov"; \
		echo "  Fedora/RHEL:   sudo dnf install lcov"; \
		echo "  macOS:         brew install lcov"; \
		echo "  Arch Linux:    sudo pacman -S lcov"; \
		exit 1; \
	fi

-include $(DEPS)

# --- libFuzzer targets (clang + -fsanitize=fuzzer) ---
FUZZ_DIR := $(BUILD_DIR)/fuzz
FUZZ_CC  := $(shell command -v clang 2>/dev/null)
FUZZ_CFLAGS := -O1 -g -Wall -Wextra -Iinclude -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined -DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
FUZZ_LDFLAGS := -fsanitize=fuzzer,address,undefined -L$(LIB_DIR) -l$(LIB_NAME) -lm -pthread -Wl,-rpath,$(abspath $(LIB_DIR))

.PHONY: fuzz fuzz-run fuzz-corpus
fuzz: lib
ifndef FUZZ_CC
	@echo "clang not found; install clang to build fuzz targets"
	@exit 1
endif
	@mkdir -p $(FUZZ_DIR)
	$(FUZZ_CC) $(FUZZ_CFLAGS) tests/fuzz/fuzz_wal_apply.c $(FUZZ_LDFLAGS) -o $(FUZZ_DIR)/fuzz_wal_apply
	$(FUZZ_CC) $(FUZZ_CFLAGS) tests/fuzz/fuzz_grpc_decode.c $(FUZZ_LDFLAGS) -o $(FUZZ_DIR)/fuzz_grpc_decode
	$(FUZZ_CC) $(FUZZ_CFLAGS) tests/fuzz/fuzz_wal_replay.c $(FUZZ_LDFLAGS) -o $(FUZZ_DIR)/fuzz_wal_replay
	$(FUZZ_CC) $(FUZZ_CFLAGS) tests/fuzz/fuzz_repl_frame.c $(FUZZ_LDFLAGS) -o $(FUZZ_DIR)/fuzz_repl_frame
	$(FUZZ_CC) $(FUZZ_CFLAGS) tests/fuzz/fuzz_grpc_frame.c $(FUZZ_LDFLAGS) -o $(FUZZ_DIR)/fuzz_grpc_frame
	$(FUZZ_CC) $(FUZZ_CFLAGS) tests/fuzz/fuzz_grpc_dispatch.c $(FUZZ_LDFLAGS) -o $(FUZZ_DIR)/fuzz_grpc_dispatch
	$(FUZZ_CC) $(FUZZ_CFLAGS) tests/fuzz/fuzz_posting_segment.c $(FUZZ_LDFLAGS) -o $(FUZZ_DIR)/fuzz_posting_segment
	$(FUZZ_CC) $(FUZZ_CFLAGS) tests/fuzz/fuzz_json.c $(FUZZ_LDFLAGS) -o $(FUZZ_DIR)/fuzz_json
	$(FUZZ_CC) $(FUZZ_CFLAGS) tests/fuzz/fuzz_filter_expr.c $(FUZZ_LDFLAGS) -o $(FUZZ_DIR)/fuzz_filter_expr
	$(FUZZ_CC) $(FUZZ_CFLAGS) tests/fuzz/fuzz_sql.c $(FUZZ_LDFLAGS) -o $(FUZZ_DIR)/fuzz_sql
	$(FUZZ_CC) $(FUZZ_CFLAGS) tests/fuzz/fuzz_cypher.c $(FUZZ_LDFLAGS) -o $(FUZZ_DIR)/fuzz_cypher
	@echo "Built fuzzers in $(FUZZ_DIR)"

fuzz-run: export LSAN_OPTIONS = suppressions=$(abspath tests/fuzz/lsan.supp)
fuzz-run: fuzz
	@echo "Running fuzz_wal_apply..."
	@$(FUZZ_DIR)/fuzz_wal_apply tests/fuzz/corpus/wal -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_grpc_decode..."
	@$(FUZZ_DIR)/fuzz_grpc_decode tests/fuzz/corpus/grpc -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_grpc_frame..."
	@$(FUZZ_DIR)/fuzz_grpc_frame tests/fuzz/corpus/grpc -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_grpc_dispatch..."
	@$(FUZZ_DIR)/fuzz_grpc_dispatch tests/fuzz/corpus/grpc -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_repl_frame..."
	@$(FUZZ_DIR)/fuzz_repl_frame tests/fuzz/corpus/repl -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_posting_segment..."
	@mkdir -p tests/fuzz/corpus/posting
	@$(FUZZ_DIR)/fuzz_posting_segment tests/fuzz/corpus/posting -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_wal_replay (empty seed corpus; full-file replay)..."
	@mkdir -p tests/fuzz/corpus/wal_replay_empty
	@$(FUZZ_DIR)/fuzz_wal_replay tests/fuzz/corpus/wal_replay_empty -runs=5000 -max_len=8192 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_json..."
	@mkdir -p tests/fuzz/corpus/json
	@$(FUZZ_DIR)/fuzz_json tests/fuzz/corpus/json -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_filter_expr..."
	@mkdir -p tests/fuzz/corpus/filter_expr
	@$(FUZZ_DIR)/fuzz_filter_expr tests/fuzz/corpus/filter_expr -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_sql..."
	@mkdir -p tests/fuzz/corpus/sql
	@$(FUZZ_DIR)/fuzz_sql tests/fuzz/corpus/sql -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_cypher..."
	@mkdir -p tests/fuzz/corpus/cypher
	@$(FUZZ_DIR)/fuzz_cypher tests/fuzz/corpus/cypher -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1

fuzz-corpus: lib
	@bash tests/fuzz/gen_corpus.sh
