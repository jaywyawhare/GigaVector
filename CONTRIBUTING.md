# Contributing to GigaVector

Thank you for your interest in contributing to GigaVector! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)
8. [Issue Reporting](#issue-reporting)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

### Unacceptable Behavior

- Harassment or discriminatory language
- Personal attacks
- Trolling or inflammatory comments

---

## Getting Started

### Prerequisites

- C compiler (GCC 9+ or Clang 10+)
- Make or CMake
- Git
- Basic understanding of C programming

### Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/GigaVector.git
cd GigaVector

# Add upstream remote
git remote add upstream https://github.com/jaywyawhare/GigaVector.git
```

---

## Development Setup

### Building from Source

```bash
# Build the library
make clean
make lib

# Run tests
make c-test

# Build with debug symbols
make CFLAGS="-g -O0" lib
```

### Development Tools

**Recommended:**
- Editor: VS Code, Vim, or your preferred editor
- Debugger: GDB
- Memory checker: Valgrind
- Static analysis: cppcheck, clang static analyzer

**Setup:**
```bash
# Install development tools
sudo apt-get install gdb valgrind cppcheck

# Run static analysis
cppcheck --enable=all src/
```

---

## Coding Standards

### Style Guide

**Indentation:**
- Use 4 spaces (no tabs)
- Consistent indentation throughout

**Naming Conventions:**
- Functions: `snake_case` with `gv_` prefix (e.g., `gv_db_open`)
- Types: `PascalCase` with `GV_` prefix (e.g., `GV_Database`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_RESPONSE_SIZE`)
- Variables: `snake_case` (e.g., `vector_count`)

**Function Documentation:**
```c
/**
 * @brief Brief description of function.
 *
 * Longer description if needed, explaining what the function does,
 * any important details, and usage notes.
 *
 * @param param1 Description of parameter 1.
 * @param param2 Description of parameter 2.
 * @return Description of return value.
 * @retval 0 On success.
 * @retval -1 On error.
 */
int gv_example_function(int param1, const char *param2);
```

**Code Formatting:**
```c
// Good: Clear spacing and alignment
int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension) {
    if (db == NULL || data == NULL) {
        return -1;
    }
    
    // Implementation
    return 0;
}

// Bad: Inconsistent spacing
int gv_db_add_vector(GV_Database*db,const float*data,size_t dimension){
if(db==NULL||data==NULL){return -1;}
// Implementation
return 0;}
```

### Best Practices

**1. Error Handling:**
```c
// Always check return values
int result = gv_db_add_vector(db, vector, dimension);
if (result != 0) {
    fprintf(stderr, "Failed to add vector\n");
    return -1;
}
```

**2. Memory Management:**
```c
// Always free allocated memory
char *buffer = malloc(size);
if (buffer == NULL) {
    return -1;
}
// ... use buffer ...
free(buffer);
buffer = NULL;  // Prevent use-after-free
```

**3. Input Validation:**
```c
// Validate all inputs
int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension) {
    if (db == NULL || data == NULL) {
        return -1;
    }
    if (dimension == 0 || dimension != db->dimension) {
        return -1;
    }
    // ... implementation ...
}
```

**4. Thread Safety:**
```c
// Document thread safety
// This function is thread-safe for concurrent reads
// Writes require external synchronization
int gv_db_search(const GV_Database *db, ...);
```

---

## Testing

### Writing Tests

**Test Structure:**
```c
#include "gigavector/gigavector.h"
#include <assert.h>
#include <stdio.h>

void test_basic_operations(void) {
    // Setup
    GV_Database *db = gv_db_open(NULL, 4, GV_INDEX_TYPE_KDTREE);
    assert(db != NULL);
    
    // Test
    float vector[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int result = gv_db_add_vector(db, vector, 4);
    assert(result == 0);
    
    // Verify
    GV_DBStats stats;
    gv_db_get_stats(db, &stats);
    assert(stats.vector_count == 1);
    
    // Cleanup
    gv_db_close(db);
}

int main(void) {
    test_basic_operations();
    printf("All tests passed\n");
    return 0;
}
```

### Running Tests

```bash
# Run all tests
make c-test

# Run specific test
make c-test-single TEST=test_db

# Run with valgrind
make test-valgrind

# Run with sanitizers
make test-asan
```

### Test Coverage

```bash
# Generate coverage report
make test-coverage

# View HTML report
make test-coverage-html
# Open build/coverage_html/index.html
```

---

## Documentation

### Code Documentation

**Header Files:**
- Document all public functions
- Include parameter descriptions
- Document return values and error codes
- Add usage examples for complex functions

**Example:**
```c
/**
 * @brief Performs k-nearest neighbor search.
 *
 * Searches the database for the k nearest vectors to the query vector
 * using the specified distance metric.
 *
 * @param db Database handle; must be non-NULL.
 * @param query Query vector of dimension db->dimension.
 * @param k Number of results to return (must be > 0).
 * @param distance_type Distance metric (EUCLIDEAN or COSINE).
 * @param results Output array; must be pre-allocated for k elements.
 * @return Number of results found (0 to k), or -1 on error.
 *
 * @note This function is thread-safe for concurrent reads.
 * @note Results are sorted by distance (closest first).
 */
int gv_db_search(const GV_Database *db, const float *query, size_t k,
                GV_DistanceType distance_type, GV_SearchResult *results);
```

### Documentation Files

**When to update:**
- Adding new features: Update relevant docs in `docs/`
- Changing APIs: Update `docs/API_REFERENCE.md`
- Fixing bugs: Update `docs/TROUBLESHOOTING.md` if relevant

**Documentation Style:**
- Clear, concise language
- Code examples for complex concepts
- Cross-reference related topics
- Keep examples up-to-date

---

## Pull Request Process

### Before Submitting

1. **Update your fork:**
   ```bash
   git fetch upstream
   git checkout master
   git merge upstream/master
   ```

2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes:**
   - Write code following coding standards
   - Add tests for new features
   - Update documentation
   - Ensure all tests pass

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add feature: description of changes"
   ```

   **Commit Message Guidelines:**
   - Use imperative mood: "Add feature" not "Added feature"
   - First line: Brief summary (50 chars or less)
   - Body: Detailed explanation (if needed)
   - Reference issues: "Fixes #123"

5. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Pull Request Template

**Title:** Brief description of changes

**Description:**
- What changes were made?
- Why were they made?
- How were they tested?

**Checklist:**
- [ ] Code follows style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] No memory leaks (valgrind clean)
- [ ] No compiler warnings

### Review Process

1. **Automated checks** run on PR
2. **Maintainers review** code and tests
3. **Feedback provided** if changes needed
4. **Approval** when ready
5. **Merge** by maintainer

---

## Issue Reporting

### Bug Reports

**Template:**
```markdown
**Description:**
Clear description of the bug.

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Behavior:**
What should happen.

**Actual Behavior:**
What actually happens.

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- GigaVector version: [e.g., commit hash]
- Compiler: [e.g., GCC 11.2]

**Additional Context:**
Any other relevant information.
```

### Feature Requests

**Template:**
```markdown
**Feature Description:**
Clear description of the feature.

**Use Case:**
Why is this feature needed?

**Proposed Solution:**
How should it work?

**Alternatives Considered:**
Other approaches you've thought about.
```

---

## Development Workflow

### Branch Strategy

- `master`: Stable, production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `docs/*`: Documentation updates

### Release Process

1. Feature branches merged to `develop`
2. Testing and validation
3. Merge `develop` to `master`
4. Tag release version
5. Create release notes

---

## Getting Help

### Questions?

- **Documentation**: Check `docs/` directory
- **Examples**: See `docs/examples/`
- **Issues**: Search existing issues on GitHub
- **Discussions**: Use GitHub Discussions

### Communication

- Be respectful and professional
- Provide context when asking questions
- Help others when you can
- Follow the code of conduct

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Acknowledged in documentation

Thank you for contributing to GigaVector!

