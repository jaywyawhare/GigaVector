import unittest
from unittest import mock

from gigavector.retry import RetryPolicy, call_with_retry, is_retryable


class TestRetry(unittest.TestCase):
    def test_no_retry_on_value_error(self) -> None:
        calls = {"n": 0}

        def fail() -> None:
            calls["n"] += 1
            raise ValueError("bad input")

        with self.assertRaises(ValueError):
            call_with_retry(fail, RetryPolicy(max_retries=3))
        self.assertEqual(calls["n"], 1)

    def test_retries_runtime_error(self) -> None:
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return "ok"

        with mock.patch("gigavector.retry.time.sleep"):
            result = call_with_retry(flaky, RetryPolicy(max_retries=3, base_delay=0))
        self.assertEqual(result, "ok")
        self.assertEqual(calls["n"], 3)

    def test_is_retryable(self) -> None:
        self.assertFalse(is_retryable(ValueError()))
        self.assertTrue(is_retryable(RuntimeError()))


if __name__ == "__main__":
    unittest.main()
