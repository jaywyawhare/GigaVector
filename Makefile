CC      := gcc
BASE_CFLAGS := -O3 -g -Wall -Wextra -Wimplicit-fallthrough -Wformat-truncation -MMD -Iinclude -pthread -fPIC
# SIMD: default to -march=native for local builds (build host == run host), which
# activates the AVX2/FMA distance kernels in hnsw.c/distance.c/ivfpq.c (huge speedup
# for HNSW/IVF build + search). Override for portable binaries, e.g.
#   make SIMD_FLAGS="-mavx2 -mfma"   (AVX2-only; runtime-gated in-code)
#   make SIMD_FLAGS=                 (disable SIMD entirely / non-x86)
SIMD_FLAGS ?= -march=native
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
BENCH_FILES := benchmarks/benchmark_simd.c benchmarks/benchmark_compare.c benchmarks/benchmark_ivfpq.c benchmarks/benchmark_ivfpq_recall.c benchmarks/bench_ivfdisk.c benchmarks/bench_hnsw_build.c benchmarks/bench_scale.c
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

.PHONY: bench-hnsw-build
bench-hnsw-build: $(BENCH_DIR)/bench_hnsw_build
	@echo "=== HNSW build benchmark (20k x 128) ==="
	@LD_LIBRARY_PATH=$(LIB_DIR) $(BENCH_DIR)/bench_hnsw_build 20000 128

.PHONY: bench-scale
bench-scale: $(BENCH_DIR)/bench_scale
	@echo "=== Scale benchmark (1M x 128, HNSW) ==="
	@LD_LIBRARY_PATH=$(LIB_DIR) $(BENCH_DIR)/bench_scale 1000000 128 200 10

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

.PHONY: gvserver
gvserver: $(BIN_DIR)/gvserver
$(BIN_DIR)/gvserver: tools/gvserver.c $(STATIC_LIB)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) tools/gvserver.c $(STATIC_LIB) $(LDFLAGS) -o $@
	@echo "Built server daemon: $@"

.PHONY: lib
lib: $(STATIC_LIB) $(SHARED_LIB)

# Strict-warnings build: compile the library with -Werror on top of the existing
# -Wall -Wextra so any NEW warning fails the build. Opt-in only (NOT the default
# build), since pre-existing warnings elsewhere would otherwise break `make`.
# Wired into CI as its own job.
.PHONY: strict
# -Wmaybe-uninitialized is excluded from -Werror: it is a well-known false-positive
# source under -O2/-O3 (e.g. cypher.c run() where `no` is provably initialized to 0).
strict: CFLAGS += -Werror -Wno-error=maybe-uninitialized
strict:
	@$(MAKE) clean
	@$(MAKE) CFLAGS="$(CFLAGS)" lib
	@echo "Strict-warnings build passed (-Werror -Wall -Wextra)"

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

$(BENCH_DIR)/bench_hnsw_build: benchmarks/bench_hnsw_build.c $(STATIC_LIB)
	@mkdir -p $(BENCH_DIR)
	$(CC) $(CFLAGS) $< -L$(LIB_DIR) -l$(LIB_NAME) $(LDFLAGS) -o $@
	@echo "Built benchmark: $@"

$(BENCH_DIR)/bench_scale: benchmarks/bench_scale.c $(STATIC_LIB)
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
	@$(MAKE) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" lib
	@$(MAKE) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" $(TEST_BINS)
	@echo "Running tests with AddressSanitizer..."
	@for test in $(TEST_BINS); do \
		echo "Running $$test with ASAN..."; \
		ASAN_OPTIONS=detect_leaks=1 \
		LD_LIBRARY_PATH=$(LIB_DIR):$$LD_LIBRARY_PATH $$test || exit 1; \
	done
	@echo "All ASAN tests passed"

.PHONY: test-tsan
test-tsan: CFLAGS += $(TSAN_FLAGS)
test-tsan: LDFLAGS += -fsanitize=thread
test-tsan:
	@$(MAKE) clean
	@$(MAKE) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" lib
	@$(MAKE) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" $(TEST_BINS)
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
	@$(MAKE) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" lib
	@$(MAKE) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" $(TEST_BINS)
	@echo "Running tests with UndefinedBehaviorSanitizer..."
	@for test in $(TEST_BINS); do \
		echo "Running $$test with UBSAN..."; \
		UBSAN_OPTIONS=halt_on_error=1 \
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
#   test_flat / test_corrupt_resilience are clean and gated:
#     - test_flat: tests free their owned search results (gv_search_results_free).
#     - test_corrupt_resilience: the db_open_from_memory corrupt/truncated-input
#       error paths now fully tear down the partially-built db.
#   test_db / test_hnsw are now gated too: the former db_add_vector HNSW-family
#   GV_Vector-shell leak (data copied into soa_storage but the shell never freed
#   on success for the HNSW/IVFPQ/IVFFLAT/IVFSQ8/IVFTURBOQUANT index families) is
#   FIXED, and the whole suite is valgrind-clean.
VALGRIND_CORE_TESTS := \
	index/test_flat \
	storage/test_corrupt_resilience \
	storage/test_db \
	index/test_hnsw \
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

# test-all is a superset of everything CI runs: the C/Python suites, all three
# sanitizers, the DST oracles, corrupt-input resilience, and the curated
# valgrind gate. A failure in any of these fails the target (valgrind is no
# longer swallowed).
.PHONY: test-all
test-all: c-test python-test-comprehensive test-asan test-tsan test-ubsan \
          dst-test test-corrupt-resilience test-valgrind-core
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
	$(FUZZ_CC) $(FUZZ_CFLAGS) -D_GNU_SOURCE tests/fuzz/fuzz_graph_wal.c $(FUZZ_LDFLAGS) -o $(FUZZ_DIR)/fuzz_graph_wal
	@echo "Built fuzzers in $(FUZZ_DIR)"

# Crash / leak / oom / timeout reproducers land under build/ (via
# -artifact_prefix) instead of polluting the repo root.
FUZZ_CRASH_DIR := $(FUZZ_DIR)/crashes
FUZZ_ARTIFACT_PREFIX := -artifact_prefix=$(FUZZ_CRASH_DIR)/

# CORPUS_CACHE: optional writable directory placed FIRST in each fuzzer's
# corpus list. libFuzzer writes every newly discovered unit there while still
# reading the in-repo seed corpus that follows it, so CI can cache the dir and
# coverage accumulates across runs instead of resetting to the seeds each time.
# Unset (the default) keeps the historical behaviour: seeds only, new units
# discarded with the workspace.
CORPUS_CACHE ?=
FUZZ_CORPUS_ARGS := $(if $(CORPUS_CACHE),$(CORPUS_CACHE),)

fuzz-run: export LSAN_OPTIONS = suppressions=$(abspath tests/fuzz/lsan.supp)
fuzz-run: fuzz
	@mkdir -p $(FUZZ_CRASH_DIR) $(CORPUS_CACHE)
	@echo "Running fuzz_wal_apply..."
	@$(FUZZ_DIR)/fuzz_wal_apply $(FUZZ_CORPUS_ARGS) tests/fuzz/corpus/wal $(FUZZ_ARTIFACT_PREFIX) -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_grpc_decode..."
	@$(FUZZ_DIR)/fuzz_grpc_decode $(FUZZ_CORPUS_ARGS) tests/fuzz/corpus/grpc $(FUZZ_ARTIFACT_PREFIX) -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_grpc_frame..."
	@$(FUZZ_DIR)/fuzz_grpc_frame $(FUZZ_CORPUS_ARGS) tests/fuzz/corpus/grpc $(FUZZ_ARTIFACT_PREFIX) -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_grpc_dispatch..."
	@$(FUZZ_DIR)/fuzz_grpc_dispatch $(FUZZ_CORPUS_ARGS) tests/fuzz/corpus/grpc $(FUZZ_ARTIFACT_PREFIX) -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_repl_frame..."
	@$(FUZZ_DIR)/fuzz_repl_frame $(FUZZ_CORPUS_ARGS) tests/fuzz/corpus/repl $(FUZZ_ARTIFACT_PREFIX) -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_posting_segment..."
	@mkdir -p tests/fuzz/corpus/posting
	@$(FUZZ_DIR)/fuzz_posting_segment $(FUZZ_CORPUS_ARGS) tests/fuzz/corpus/posting $(FUZZ_ARTIFACT_PREFIX) -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_wal_replay (empty seed corpus; full-file replay)..."
	@mkdir -p tests/fuzz/corpus/wal_replay_empty
	@$(FUZZ_DIR)/fuzz_wal_replay $(FUZZ_CORPUS_ARGS) tests/fuzz/corpus/wal_replay_empty $(FUZZ_ARTIFACT_PREFIX) -runs=5000 -max_len=8192 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_json..."
	@mkdir -p tests/fuzz/corpus/json
	@$(FUZZ_DIR)/fuzz_json $(FUZZ_CORPUS_ARGS) tests/fuzz/corpus/json $(FUZZ_ARTIFACT_PREFIX) -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_filter_expr..."
	@mkdir -p tests/fuzz/corpus/filter_expr
	@$(FUZZ_DIR)/fuzz_filter_expr $(FUZZ_CORPUS_ARGS) tests/fuzz/corpus/filter_expr $(FUZZ_ARTIFACT_PREFIX) -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_sql..."
	@mkdir -p tests/fuzz/corpus/sql
	@$(FUZZ_DIR)/fuzz_sql $(FUZZ_CORPUS_ARGS) tests/fuzz/corpus/sql $(FUZZ_ARTIFACT_PREFIX) -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@echo "Running fuzz_cypher..."
	@mkdir -p tests/fuzz/corpus/cypher
	@echo "Running fuzz_graph_wal..."
	@$(FUZZ_DIR)/fuzz_graph_wal $(FUZZ_CORPUS_ARGS) $(FUZZ_ARTIFACT_PREFIX) -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1
	@$(FUZZ_DIR)/fuzz_cypher $(FUZZ_CORPUS_ARGS) tests/fuzz/corpus/cypher $(FUZZ_ARTIFACT_PREFIX) -max_total_time=30 -rss_limit_mb=512 -print_final_stats=1

fuzz-corpus: lib
	@bash tests/fuzz/gen_corpus.sh

# --- Formal verification: Quint specs, CBMC harnesses, structural checks -----
#
# Specs live in specs/quint/*.qnt and are globbed, so a new spec cannot drift
# out of CI. See docs/formal_verification_plan.md.
#
# Cost tiers, cheapest first:
#   formal-lint      lock-order static check            -- no toolchain, ~1s
#   quint-typecheck  type + effect check, every spec    -- seconds
#   quint-test       run scenarios / regression witness -- seconds
#   cbmc-smoke       compile+run harnesses concretely   -- seconds
#   quint-replay     ITF traces vs. the real C          -- seconds
#   quint-verify     Apalache symbolic model checking   -- minutes (nightly)
#   cbmc             bounded proof over the C           -- minutes (nightly)
QUINT       ?= quint
QUINT_SPECS := $(sort $(wildcard specs/quint/*.qnt))

# Every module that declares a `run`. "file:module".
QUINT_TEST_MODULES := \
  specs/quint/raft.qnt:raft3 \
  specs/quint/raft.qnt:raft5_prefix \
  specs/quint/mvcc.qnt:mvcc \
  specs/quint/consistency.qnt:consistency \
  specs/quint/wal_recovery.qnt:wal_recovery \
  specs/quint/wal_recovery.qnt:wal_group_commit \
  specs/quint/vlog_gc.qnt:vlog_gc \
  specs/quint/quota.qnt:quota \
  specs/quint/ttl.qnt:ttl \
  specs/quint/shard_rebalance.qnt:shard_rebalance \
  specs/quint/replication.qnt:replication \
  specs/quint/lock_order.qnt:lock_order \
  specs/quint/cdc.qnt:cdc \
  specs/quint/tiered_storage.qnt:tiered_storage \
  specs/quint/index_swap.qnt:index_swap \
  specs/quint/rbac.qnt:rbac \
  specs/quint/cluster.qnt:cluster \
  specs/quint/snapshot.qnt:snapshot \
  specs/quint/cache.qnt:cache \
  specs/quint/cache.qnt:cache_invalidating \
  specs/quint/namespace.qnt:namespace \
  specs/quint/versioning.qnt:versioning \
  specs/quint/webhook.qnt:webhook \
  specs/quint/auth.qnt:auth \
  specs/quint/dedup.qnt:dedup \
  specs/quint/scroll_cursor.qnt:scroll_cursor \
  specs/quint/memory_layer.qnt:memory_layer \
  specs/quint/bulk_import.qnt:bulk_import \
  specs/quint/ab_test.qnt:ab_test \
  specs/quint/repl_wal_crash.qnt:repl_wal_crash

# Model-checking runs: "file:module:invariant:max-steps".
# ttl and quota are deliberately absent: their `safety` invariant is currently
# FALSE by design (confirmed TOCTOU findings, see docs/formal_verification_plan.md).
# Re-add them here once those are fixed, so the fix cannot silently regress.
QUINT_CHECKS := \
  specs/quint/raft.qnt:raft3:safety:6 \
  specs/quint/mvcc.qnt:mvcc:safety:8 \
  specs/quint/consistency.qnt:consistency:safety:8 \
  specs/quint/wal_recovery.qnt:wal_recovery:safety:8 \
  specs/quint/vlog_gc.qnt:vlog_gc:safety:8 \
  specs/quint/lock_order.qnt:lock_order:safety:8 \
  specs/quint/snapshot.qnt:snapshot \
  specs/quint/cache.qnt:cache \
  specs/quint/cache.qnt:cache_invalidating \
  specs/quint/namespace.qnt:namespace \
  specs/quint/versioning.qnt:versioning \
  specs/quint/webhook.qnt:webhook \
  specs/quint/auth.qnt:auth \
  specs/quint/dedup.qnt:dedup \
  specs/quint/scroll_cursor.qnt:scroll_cursor \
  specs/quint/memory_layer.qnt:memory_layer \
  specs/quint/bulk_import.qnt:bulk_import \
  specs/quint/ab_test.qnt:ab_test:safety:8 \
  specs/quint/repl_wal_crash.qnt:repl_wal_crash:safety:6

.PHONY: formal-lint
formal-lint:
	@python3 scripts/check_lock_order.py

# Every .c file under src/ must carry a verification classification, and every
# classification must point at an artifact that exists. This is what stops new
# code from silently escaping verification, and stops the manifest from
# over-claiming coverage it does not have.
.PHONY: formal-coverage
formal-coverage:
	@python3 scripts/check_coverage.py

.PHONY: quint-typecheck
quint-typecheck:
	@command -v $(QUINT) >/dev/null 2>&1 || { \
	  echo "quint not found; install with: npm install -g @informalsystems/quint"; exit 1; }
	@for s in $(QUINT_SPECS); do \
	  printf '  typecheck %-36s' $$s; \
	  $(QUINT) typecheck $$s >/dev/null || { echo FAIL; $(QUINT) typecheck $$s; exit 1; }; \
	  echo ok; \
	done
	@echo "All Quint specs typecheck."

.PHONY: quint-test
quint-test:
	@for c in $(QUINT_TEST_MODULES); do \
	  f=$$(echo $$c | cut -d: -f1); m=$$(echo $$c | cut -d: -f2); \
	  printf '  test %-22s' $$m; \
	  $(QUINT) test --main=$$m $$f >/dev/null 2>&1 || { \
	    echo FAIL; $(QUINT) test --main=$$m $$f; exit 1; }; \
	  echo ok; \
	done
	@echo "All Quint scenario tests pass."

# Vacuity gate. A spec whose `step` can never fire passes every invariant
# trivially -- the worst failure mode for a spec suite, because it looks green.
# Each entry names a state the model MUST be able to reach; if it becomes
# unreachable the spec has died and this fails.
# "file:module:witness-expression".
QUINT_WITNESSES := \
  'specs/quint/raft.qnt:raft3:role.get(0) == "Leader"' \
  'specs/quint/mvcc.qnt:mvcc:versions.size() > 1' \
  'specs/quint/consistency.qnt:consistency:leaderPos > 1' \
  'specs/quint/wal_recovery.qnt:wal_recovery:durable.length() > 1' \
  'specs/quint/vlog_gc.qnt:vlog_gc:log.length() > 1' \
  'specs/quint/quota.qnt:quota:currentVectors > 0' \
  'specs/quint/ttl.qnt:ttl:pendingDelete.size() > 0' \
  'specs/quint/shard_rebalance.qnt:shard_rebalance:holders.get(1).size() > 1' \
  'specs/quint/replication.qnt:replication:leaderLog > 1' \
  'specs/quint/lock_order.qnt:lock_order:held.get(1).size() > 1' \
  'specs/quint/cdc.qnt:cdc:published.length() > 1' \
  'specs/quint/tiered_storage.qnt:tiered_storage:tier.get(1) == "hot"' \
  'specs/quint/index_swap.qnt:index_swap:stored.size() > 1' \
  'specs/quint/rbac.qnt:rbac:effectivePerms(1).size() > 0' \
  'specs/quint/cluster.qnt:cluster:members.size() > 2' \
  'specs/quint/snapshot.qnt:snapshot:haveSnapshot' \
  'specs/quint/cache.qnt:cache:servedStale.size() > 0' \
  'specs/quint/namespace.qnt:namespace:aliases.size() > 0' \
  'specs/quint/versioning.qnt:versioning:versions.size() > 1' \
  'specs/quint/webhook.qnt:webhook:delivered.size() > 0' \
  'specs/quint/auth.qnt:auth:accepted.size() > 0' \
  'specs/quint/dedup.qnt:dedup:stored.size() > 1' \
  'specs/quint/scroll_cursor.qnt:scroll_cursor:returned.length() > 1' \
  'specs/quint/memory_layer.qnt:memory_layer:memories.size() > 1' \
  'specs/quint/bulk_import.qnt:bulk_import:inserted.size() > 1' \
  'specs/quint/ab_test.qnt:ab_test:recorded.size() > 0' \
  'specs/quint/repl_wal_crash.qnt:repl_wal_crash:ackedCommitted.size() > 0'

.PHONY: quint-witness
quint-witness:
	@for c in $(QUINT_WITNESSES); do \
	  f=$$(echo "$$c" | cut -d: -f1); m=$$(echo "$$c" | cut -d: -f2); \
	  w=$$(echo "$$c" | cut -d: -f3-); \
	  printf '  witness %-20s' $$m; \
	  out=$$($(QUINT) run --main=$$m --witnesses="$$w" --max-steps=10 \
	          --max-samples=500 $$f 2>&1 | grep -oE 'witnessed in [0-9]+ trace' | head -1); \
	  n=$$(echo "$$out" | grep -oE '[0-9]+'); \
	  if [ -z "$$n" ] || [ "$$n" -eq 0 ]; then \
	    echo "UNREACHABLE ($$w)"; exit 1; \
	  fi; \
	  echo "ok ($$n traces)"; \
	done
	@echo "All specs reach their witness states (no vacuous passes)."

# UNBOUNDED proofs. `quint-verify` is bounded (--max-steps): a clean run means
# no counterexample within the bound. An inductive invariant is stronger --
# it holds in every initial state, is preserved by every step, and implies the
# target, so the result holds for ALL reachable states with no bound at all.
#
# Each entry needs a hand-written `typeOk` (domain constraints via `.in(<set>)`,
# because the checker enumerates candidate states from the predicate) plus
# whatever strengthening makes the conjunction closed under `step`. That is real
# work per spec, which is why this list is shorter than QUINT_CHECKS.
QUINT_INDUCTIVE := \
  specs/quint/auth.qnt:auth \
  specs/quint/lock_order.qnt:lock_order \
  specs/quint/versioning.qnt:versioning \
  specs/quint/cluster.qnt:cluster \
  specs/quint/bulk_import.qnt:bulk_import \
  specs/quint/namespace.qnt:namespace

.PHONY: quint-induct
quint-induct:
	@for c in $(QUINT_INDUCTIVE); do \
	  f=$$(echo $$c | cut -d: -f1); m=$$(echo $$c | cut -d: -f2); \
	  printf '  induct %-20s' $$m; \
	  $(QUINT) verify --main=$$m --inductive-invariant=indInv \
	    --invariant=safety $$f >/dev/null 2>&1 \
	    && echo "PROVED (unbounded)" \
	    || { echo FAIL; exit 1; }; \
	done
	@echo "Inductive invariants established -- these hold for ALL reachable states."

# BOUNDED model checking via Apalache. scripts/run_quint_verify.py tunes
# --max-steps per spec (state size varies enormously) and applies a per-check
# timeout, so a check that cannot finish is reported INTRACTABLE rather than
# hanging while the summary still reads green.
QUINT_VERIFY_TIMEOUT ?= 600

.PHONY: quint-verify
quint-verify:
	@python3 -u scripts/run_quint_verify.py --quint $(QUINT) \
	  --timeout $(QUINT_VERIFY_TIMEOUT)

# Replay Quint-generated ITF traces against the real C implementation.
# Traces under specs/quint/traces/ whose name contains "conformance" must match
# exactly; the *-violation / *-double-count witnesses are expected to diverge
# (they were generated from the PRE-FIX model and are kept as regressions).
# Dispatch by filename prefix: raft* traces drive src/admin/raft.c via
# replay_raft.py; wal-recovery* traces drive src/storage/wal.c (crash +
# recovery against a real file) via replay_wal.py.
.PHONY: quint-replay
quint-replay: lib
	@for t in specs/quint/traces/*conformance*.itf.json; do \
	  [ -e "$$t" ] || continue; \
	  case "$$t" in \
	    *wal-recovery*) python3 specs/replay/replay_wal.py $$t || exit 1 ;; \
	    *)              python3 specs/replay/replay_raft.py $$t || exit 1 ;; \
	  esac || exit 1; \
	done
	@echo "ITF trace replay conformant."

.PHONY: cbmc-smoke
cbmc-smoke: lib
	@for h in specs/cbmc/harness_*.c; do \
	  b=$(BUILD_DIR)/cbmc-$$(basename $$h .c); \
	  printf '  %-40s' $$h; \
	  $(CC) -O0 -g -Wall -Wextra -Iinclude $$h -o $$b $(LIB_DIR)/lib$(LIB_NAME).a \
	    -lm -pthread >/dev/null 2>&1 || { echo "BUILD FAIL"; exit 1; }; \
	  $$b >/dev/null 2>&1 || { echo "RUN FAIL"; exit 1; }; \
	  echo ok; \
	done
	@echo "CBMC harnesses compile and pass concretely."

# Real bounded proof over the C. Distinct from cbmc-smoke, which only compiles
# and concretely runs each harness once -- a single execution is not
# verification. scripts/run_cbmc.py reads each harness's CBMC-SOURCES /
# CBMC-UNWIND directives and reports PROVED / FAILED / INTRACTABLE honestly;
# INTRACTABLE harnesses are NOT proofs and the coverage manifest says so.
CBMC ?= cbmc
CBMC_TIMEOUT ?= 300

.PHONY: cbmc
cbmc:
	@command -v $(CBMC) >/dev/null 2>&1 || { \
	  echo "cbmc not found; skipping bounded proofs."; \
	  echo "install: https://github.com/diffblue/cbmc/releases"; exit 0; }
	@python3 -u scripts/run_cbmc.py --cbmc $(CBMC) --timeout $(CBMC_TIMEOUT)

.PHONY: formal
formal: formal-lint formal-coverage quint-typecheck quint-test quint-witness quint-induct cbmc-smoke
	@echo "Formal checks (fast tier) passed."
