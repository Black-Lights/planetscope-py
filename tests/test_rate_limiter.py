#!/usr/bin/env python3
"""Tests for rate limiting and retry logic.

Comprehensive test suite for rate_limiter.py functionality including
rate limiting behavior, retry mechanisms, and circuit breaker patterns.
"""

import time
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta, timezone

import requests
from requests.exceptions import ConnectionError, Timeout

from planetscope_py.rate_limiter import RateLimiter, RetryableSession, CircuitBreaker
from planetscope_py.exceptions import RateLimitError, APIError


class TestRateLimiter:
    """Test suite for RateLimiter class."""

    @pytest.fixture
    def rate_limiter(self):
        """Create RateLimiter instance for testing."""
        rates = {"search": 2, "activate": 1, "download": 3, "general": 1}
        return RateLimiter(rates=rates)

    @pytest.fixture
    def mock_session(self):
        """Create mock session for testing."""
        session = Mock()
        response = Mock()
        response.status_code = 200
        response.headers = {}
        session.request.return_value = response
        return session

    def test_initialization(self, rate_limiter):
        """Test RateLimiter initialization."""
        assert rate_limiter.rates["search"] == 2
        assert rate_limiter.rates["activate"] == 1
        assert rate_limiter.rates["download"] == 3
        assert rate_limiter.rates["general"] == 1

        assert rate_limiter.rate_limit_window == 60
        assert "initial_delay" in rate_limiter.backoff_config
        assert "max_delay" in rate_limiter.backoff_config
        assert "multiplier" in rate_limiter.backoff_config

    def test_classify_endpoint(self, rate_limiter):
        """Test endpoint classification for rate limiting."""
        # Search endpoints
        assert (
            rate_limiter._classify_endpoint(
                "https://api.planet.com/data/v1/quick-search"
            )
            == "search"
        )

        assert (
            rate_limiter._classify_endpoint(
                "https://api.planet.com/data/v1/searches/12345"
            )
            == "search"
        )

        assert (
            rate_limiter._classify_endpoint("https://api.planet.com/data/v1/stats")
            == "search"
        )

        # Activation endpoints
        assert (
            rate_limiter._classify_endpoint(
                "https://api.planet.com/data/v1/item-types/PSScene/items/123/assets/ortho/activate"
            )
            == "activate"
        )

        # Download endpoints
        assert (
            rate_limiter._classify_endpoint(
                "https://api.planet.com/data/v1/download?location=test"
            )
            == "download"
        )

        assert (
            rate_limiter._classify_endpoint(
                "https://api.planet.com/data/v1/assets/123/location"
            )
            == "download"
        )

        # General endpoints
        assert (
            rate_limiter._classify_endpoint("https://api.planet.com/data/v1/item-types")
            == "general"
        )

    def test_enforce_rate_limit_basic(self, rate_limiter, mock_session):
        """Test basic rate limit enforcement."""
        rate_limiter.session = mock_session

        # Set very low rate limit for testing
        rate_limiter.rates["general"] = 1
        rate_limiter.rate_limit_window = 1  # 1 second window

        start_time = time.time()

        # Make two requests rapidly
        rate_limiter.make_request("GET", "https://api.planet.com/data/v1/item-types")
        rate_limiter.make_request("GET", "https://api.planet.com/data/v1/item-types")

        elapsed = time.time() - start_time

        # Second request should be delayed by rate limiting
        assert elapsed >= 1.0

    def test_make_request_success(self, rate_limiter, mock_session):
        """Test successful request execution."""
        rate_limiter.session = mock_session

        response = rate_limiter.make_request(
            "GET", "https://api.planet.com/data/v1/item-types"
        )

        assert response.status_code == 200
        mock_session.request.assert_called_once()

    def test_make_request_with_429_retry(self, rate_limiter):
        """Test handling of 429 rate limit responses."""
        mock_session = Mock()

        # First call returns 429, second succeeds
        response_429 = Mock()
        response_429.status_code = 429
        response_429.headers = {"Retry-After": "2"}

        response_200 = Mock()
        response_200.status_code = 200
        response_200.headers = {}

        mock_session.request.side_effect = [response_429, response_200]
        rate_limiter.session = mock_session

        with patch("time.sleep") as mock_sleep:
            response = rate_limiter.make_request(
                "GET", "https://api.planet.com/data/v1/quick-search"
            )

            # Should have slept for retry-after duration
            mock_sleep.assert_called_with(2.0)
            assert response.status_code == 200

    def test_make_request_429_max_retries_exceeded(self, rate_limiter):
        """Test 429 handling when max retries exceeded."""
        mock_session = Mock()

        # Always return 429
        response_429 = Mock()
        response_429.status_code = 429
        response_429.headers = {"Retry-After": "1"}
        mock_session.request.return_value = response_429

        rate_limiter.session = mock_session

        with patch("time.sleep"):
            with pytest.raises(RateLimitError) as exc_info:
                rate_limiter.make_request(
                    "GET", "https://api.planet.com/data/v1/quick-search", max_retries=2
                )

            assert "Rate limit exceeded and max retries reached" in str(exc_info.value)
            assert exc_info.value.details["retry_after"] == 1.0

    def test_make_request_server_error_retry(self, rate_limiter):
        """Test retry on server errors."""
        mock_session = Mock()

        # First call returns 500, second succeeds
        response_500 = Mock()
        response_500.status_code = 500

        response_200 = Mock()
        response_200.status_code = 200
        response_200.headers = {}

        mock_session.request.side_effect = [response_500, response_200]
        rate_limiter.session = mock_session

        with patch("time.sleep") as mock_sleep:
            response = rate_limiter.make_request(
                "GET", "https://api.planet.com/data/v1/item-types"
            )

            # Should have slept for backoff
            assert mock_sleep.called
            assert response.status_code == 200

    def test_make_request_connection_error_retry(self, rate_limiter):
        """Test retry on connection errors."""
        mock_session = Mock()

        # First call raises ConnectionError, second succeeds
        response_200 = Mock()
        response_200.status_code = 200
        response_200.headers = {}

        mock_session.request.side_effect = [
            ConnectionError("Connection failed"),
            response_200,
        ]
        rate_limiter.session = mock_session

        with patch("time.sleep") as mock_sleep:
            response = rate_limiter.make_request(
                "GET", "https://api.planet.com/data/v1/item-types"
            )

            assert mock_sleep.called
            assert response.status_code == 200

    def test_make_request_max_retries_exceeded(self, rate_limiter):
        """Test behavior when max retries exceeded."""
        mock_session = Mock()
        mock_session.request.side_effect = ConnectionError("Persistent error")
        rate_limiter.session = mock_session

        with patch("time.sleep"):
            with pytest.raises(APIError) as exc_info:
                rate_limiter.make_request(
                    "GET", "https://api.planet.com/data/v1/item-types", max_retries=1
                )

            assert "Request failed after 2 attempts" in str(exc_info.value)

    def test_parse_retry_after_seconds(self, rate_limiter):
        """Test parsing Retry-After header as seconds."""
        response = Mock()
        response.headers = {"Retry-After": "30"}

        retry_after = rate_limiter._parse_retry_after(response)
        assert retry_after == 30.0

    def test_parse_retry_after_http_date(self, rate_limiter):
        """Test parsing Retry-After header as HTTP date."""
        response = Mock()
        # Set retry time to 30 seconds from now
        # FIXED: Use timezone-aware datetime instead of deprecated utcnow()
        retry_time = datetime.now(timezone.utc) + timedelta(seconds=30)
        response.headers = {
            "Retry-After": retry_time.strftime("%a, %d %b %Y %H:%M:%S GMT")
        }

        retry_after = rate_limiter._parse_retry_after(response)
        # Should be approximately 30 seconds (allow some variance for execution time)
        assert 25 <= retry_after <= 35

    def test_parse_retry_after_invalid(self, rate_limiter):
        """Test parsing invalid Retry-After header."""
        response = Mock()
        response.headers = {"Retry-After": "invalid"}

        retry_after = rate_limiter._parse_retry_after(response)
        # Should return default initial delay
        assert retry_after == rate_limiter.backoff_config["initial_delay"]

    def test_calculate_backoff_delay(self, rate_limiter):
        """Test exponential backoff delay calculation."""
        # Test increasing delays
        delay_0 = rate_limiter._calculate_backoff_delay(0)
        delay_1 = rate_limiter._calculate_backoff_delay(1)
        delay_2 = rate_limiter._calculate_backoff_delay(2)

        assert delay_0 < delay_1 < delay_2
        assert delay_2 <= rate_limiter.backoff_config["max_delay"]

    def test_track_request_history(self, rate_limiter, mock_session):
        """Test request history tracking."""
        rate_limiter.session = mock_session

        # Make several requests
        for i in range(5):
            rate_limiter.make_request(
                "GET", "https://api.planet.com/data/v1/item-types"
            )

        # Check history
        history = rate_limiter.request_history["general"]
        assert len(history) == 5

        # Each entry should have timestamp and duration
        for timestamp, duration in history:
            assert isinstance(timestamp, float)
            assert isinstance(duration, float)
            assert duration >= 0

    def test_get_current_rate_status(self, rate_limiter, mock_session):
        """Test getting current rate limiting status."""
        rate_limiter.session = mock_session

        # Make some requests
        for i in range(3):
            rate_limiter.make_request(
                "GET", "https://api.planet.com/data/v1/quick-search"
            )

        status = rate_limiter.get_current_rate_status()

        assert "search" in status
        search_status = status["search"]
        assert "limit" in search_status
        assert "current_rate" in search_status
        assert "recent_requests" in search_status
        assert "capacity_used" in search_status

        assert search_status["limit"] == 2
        assert search_status["recent_requests"] == 3

    def test_wait_for_capacity(self, rate_limiter):
        """Test waiting for rate limit capacity."""
        # Fill up the rate limit
        rate_limiter.request_history["general"] = [
            (time.time(), 0.1) for _ in range(10)
        ]

        with patch("time.sleep") as mock_sleep:
            rate_limiter.wait_for_capacity("general", required_capacity=0.5)

            # Should have waited
            assert mock_sleep.called

    def test_update_rate_limits_from_response(self, rate_limiter):
        """Test updating rate limits from response headers."""
        response = Mock()
        response.headers = {
            "X-RateLimit-Limit": "10",
            "X-RateLimit-Remaining": "5",
            "X-RateLimit-Reset": str(int(time.time()) + 3600),
        }

        rate_limiter._update_rate_limits_from_response(response, "search")

        assert rate_limiter.detected_limits["search"] == 10
        assert "search" in rate_limiter.rate_limit_reset_times

    def test_reset_rate_limits(self, rate_limiter):
        """Test resetting rate limiting state."""
        # Add some history and detected limits
        rate_limiter.request_history["search"].append((time.time(), 0.1))
        rate_limiter.detected_limits["search"] = 5
        rate_limiter.rate_limit_reset_times["search"] = time.time()

        rate_limiter.reset_rate_limits()

        assert len(rate_limiter.request_history["search"]) == 0
        assert "search" not in rate_limiter.detected_limits
        assert "search" not in rate_limiter.rate_limit_reset_times

    def test_update_rate_limit(self, rate_limiter):
        """Test updating rate limit for endpoint type."""
        original_limit = rate_limiter.rates["search"]

        rate_limiter.update_rate_limit("search", 5)

        assert rate_limiter.rates["search"] == 5
        assert rate_limiter.rates["search"] != original_limit

    def test_get_performance_metrics(self, rate_limiter, mock_session):
        """Test getting performance metrics."""
        rate_limiter.session = mock_session

        # Make some requests with varying response times
        mock_session.request.side_effect = [
            Mock(status_code=200, headers={}),
            Mock(status_code=200, headers={}),
            Mock(status_code=200, headers={}),
        ]

        # Make requests
        rate_limiter.make_request("GET", "https://api.planet.com/data/v1/quick-search")
        rate_limiter.make_request("GET", "https://api.planet.com/data/v1/quick-search")
        rate_limiter.make_request("GET", "https://api.planet.com/data/v1/item-types")

        metrics = rate_limiter.get_performance_metrics()

        assert "total_requests" in metrics
        assert "average_response_time" in metrics
        assert "endpoint_metrics" in metrics

        assert metrics["total_requests"] == 3
        assert "search" in metrics["endpoint_metrics"]
        assert "general" in metrics["endpoint_metrics"]


@pytest.fixture
def retryable_session():  # ✅ CORRECT: outside class, no 'self'
    """Create RetryableSession instance for testing."""
    rate_limiter = Mock()

    # Configure Mock attributes with real values to prevent Mock+int errors
    rate_limiter.max_retries = 3
    rate_limiter.multiplier = 2.0
    rate_limiter.session = Mock()

    return RetryableSession(rate_limiter=rate_limiter)


class TestRetryableSession:
    """Test suite for RetryableSession class."""

    def test_initialization(self, retryable_session):
        """Test RetryableSession initialization."""
        assert retryable_session.rate_limiter is not None
        assert retryable_session.session is not None
        assert retryable_session.circuit_breaker is not None
        assert "max_retries" in retryable_session.retry_config
        assert "retry_statuses" in retryable_session.retry_config

    def test_request_success(self, retryable_session):
        """Test successful request through retryable session."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}  # ✅ Add headers
        retryable_session.rate_limiter.make_request.return_value = mock_response

        response = retryable_session.request("GET", "https://api.planet.com/test")

        assert response.status_code == 200
        retryable_session.rate_limiter.make_request.assert_called_once()

    def test_request_with_retryable_status(self, retryable_session):
        """Test request retry on retryable status codes."""
        # First response fails, second succeeds
        fail_response = Mock()
        fail_response.status_code = 503
        fail_response.headers = {}  # ✅ Add headers
        fail_response.raise_for_status.side_effect = requests.HTTPError(
            "Service unavailable"
        )

        success_response = Mock()
        success_response.status_code = 200
        success_response.headers = {}  # ✅ Add headers

        retryable_session.rate_limiter.make_request.side_effect = [
            fail_response,
            success_response,
        ]

        with patch("time.sleep"):
            response = retryable_session.request("GET", "https://api.planet.com/test")

            assert response.status_code == 200
            assert retryable_session.rate_limiter.make_request.call_count == 2

    def test_request_max_retries_exceeded(self, retryable_session):
        """Test request failure when max retries exceeded."""
        fail_response = Mock()
        fail_response.status_code = 500
        fail_response.headers = {}  # ✅ Add headers
        fail_response.raise_for_status.side_effect = requests.HTTPError("Server error")

        retryable_session.rate_limiter.make_request.return_value = fail_response

        with patch("time.sleep"):
            with pytest.raises(requests.HTTPError):
                retryable_session.request("GET", "https://api.planet.com/test")

    def test_should_retry_exception(self, retryable_session):
        """Test exception retry logic."""
        # Rate limit errors should be retryable
        assert retryable_session._should_retry_exception(RateLimitError("Rate limited"))

        # Connection and timeout errors should be retryable
        assert retryable_session._should_retry_exception(
            ConnectionError("Connection failed")
        )
        assert retryable_session._should_retry_exception(Timeout("Request timed out"))

        # Validation errors should not be retryable
        from planetscope_py.exceptions import ValidationError

        assert not retryable_session._should_retry_exception(
            ValidationError("Invalid input")
        )

    def test_calculate_retry_delay(self, retryable_session):
        """Test retry delay calculation accounting for jitter randomness."""

        # Since the delay calculation includes random jitter, we need to test differently
        # Test 1: Verify delays are positive and reasonable
        delay_0 = retryable_session._calculate_retry_delay(0)
        delay_1 = retryable_session._calculate_retry_delay(1)
        delay_2 = retryable_session._calculate_retry_delay(2)

        # All delays should be positive
        assert delay_0 > 0
        assert delay_1 > 0
        assert delay_2 > 0

        # Delays should be reasonable (not too large)
        assert delay_0 < 60  # Max 1 minute
        assert delay_1 < 60
        assert delay_2 < 60

        # Test 2: Verify Retry-After header takes precedence
        mock_response = Mock()
        mock_response.headers = {"Retry-After": "15"}

        delay_with_header = retryable_session._calculate_retry_delay(0, mock_response)
        assert delay_with_header == 15.0

        # Test 3: Verify base exponential growth (without jitter) by mocking random
        with patch("random.uniform", return_value=0.5):  # Fixed jitter multiplier
            base_delay_0 = retryable_session._calculate_retry_delay(0)
            base_delay_1 = retryable_session._calculate_retry_delay(1)
            base_delay_2 = retryable_session._calculate_retry_delay(2)

            # With fixed jitter, should have exponential growth
            assert base_delay_0 < base_delay_1 < base_delay_2


class TestCircuitBreaker:
    """Test suite for CircuitBreaker class."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create CircuitBreaker instance for testing."""
        return CircuitBreaker(
            failure_threshold=3, recovery_timeout=10, expected_exception=APIError
        )

    def test_initialization(self, circuit_breaker):
        """Test CircuitBreaker initialization."""
        assert circuit_breaker.failure_threshold == 3
        assert circuit_breaker.recovery_timeout == 10
        assert circuit_breaker.expected_exception == APIError
        assert circuit_breaker.state == CircuitBreaker.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_successful_call_closed_state(self, circuit_breaker):
        """Test successful function call in closed state."""

        def successful_function():
            return "success"

        result = circuit_breaker.call(successful_function)

        assert result == "success"
        assert circuit_breaker.state == CircuitBreaker.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_failed_call_closed_state(self, circuit_breaker):
        """Test failed function call in closed state."""

        def failing_function():
            raise APIError("API failure")

        # Call should fail but circuit should remain closed
        with pytest.raises(APIError):
            circuit_breaker.call(failing_function)

        assert circuit_breaker.state == CircuitBreaker.CLOSED
        assert circuit_breaker.failure_count == 1

    def test_circuit_opens_after_threshold(self, circuit_breaker):
        """Test circuit opens after failure threshold reached."""

        def failing_function():
            raise APIError("API failure")

        # Fail enough times to trigger circuit opening
        for i in range(circuit_breaker.failure_threshold):
            with pytest.raises(APIError):
                circuit_breaker.call(failing_function)

        assert circuit_breaker.state == CircuitBreaker.OPEN
        assert circuit_breaker.failure_count == circuit_breaker.failure_threshold

    def test_open_circuit_fails_fast(self, circuit_breaker):
        """Test open circuit fails fast without calling function."""
        # Force circuit to open state
        circuit_breaker.state = CircuitBreaker.OPEN
        circuit_breaker.failure_count = circuit_breaker.failure_threshold
        circuit_breaker.last_failure_time = time.time()

        call_count = 0

        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"

        # Should fail fast without calling function
        with pytest.raises(APIError) as exc_info:
            circuit_breaker.call(test_function)

        assert "Circuit breaker is OPEN" in str(exc_info.value)
        assert call_count == 0  # Function should not have been called

    def test_circuit_transitions_to_half_open(self, circuit_breaker):
        """Test circuit transitions to half-open after recovery timeout."""
        # Force circuit to open state
        circuit_breaker.state = CircuitBreaker.OPEN
        circuit_breaker.failure_count = circuit_breaker.failure_threshold
        circuit_breaker.last_failure_time = time.time() - (
            circuit_breaker.recovery_timeout + 1
        )

        def successful_function():
            return "success"

        result = circuit_breaker.call(successful_function)

        assert result == "success"
        assert circuit_breaker.state == CircuitBreaker.HALF_OPEN

    def test_half_open_success_closes_circuit(self, circuit_breaker):
        """Test successful calls in half-open state close the circuit."""
        # Set to half-open state
        circuit_breaker.state = CircuitBreaker.HALF_OPEN
        circuit_breaker.success_count = 0

        def successful_function():
            return "success"

        # Make enough successful calls to close circuit
        for i in range(3):  # Required successes to close
            circuit_breaker.call(successful_function)

        assert circuit_breaker.state == CircuitBreaker.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_half_open_failure_reopens_circuit(self, circuit_breaker):
        """Test failure in half-open state reopens the circuit."""
        # Set to half-open state
        circuit_breaker.state = CircuitBreaker.HALF_OPEN
        circuit_breaker.success_count = 1

        def failing_function():
            raise APIError("API failure")

        with pytest.raises(APIError):
            circuit_breaker.call(failing_function)

        assert circuit_breaker.state == CircuitBreaker.OPEN

    def test_unexpected_exception_doesnt_trigger_circuit(self, circuit_breaker):
        """Test unexpected exceptions don't trigger circuit breaker."""

        def function_with_unexpected_error():
            raise ValueError("Unexpected error")

        # Should raise the unexpected exception without affecting circuit
        with pytest.raises(ValueError):
            circuit_breaker.call(function_with_unexpected_error)

        assert circuit_breaker.state == CircuitBreaker.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_get_state(self, circuit_breaker):
        """Test getting circuit breaker state."""
        state = circuit_breaker.get_state()

        assert "state" in state
        assert "failure_count" in state
        assert "success_count" in state
        assert "last_failure_time" in state
        assert "failure_threshold" in state
        assert "recovery_timeout" in state

        assert state["state"] == CircuitBreaker.CLOSED
        assert state["failure_count"] == 0
        assert state["failure_threshold"] == 3
        assert state["recovery_timeout"] == 10


class TestIntegrationScenarios:
    """Integration tests for combined rate limiting and retry functionality."""

    def test_rate_limiting_with_retries(self):
        """Test rate limiting combined with retry logic."""
        rates = {"test": 1}  # Very low rate for testing
        rate_limiter = RateLimiter(rates=rates)

        mock_session = Mock()
        responses = [
            Mock(status_code=500),  # First fails
            Mock(status_code=200, headers={}),  # Second succeeds
        ]
        mock_session.request.side_effect = responses
        rate_limiter.session = mock_session

        with patch("time.sleep") as mock_sleep:
            response = rate_limiter.make_request(
                "GET", "https://api.planet.com/test/endpoint"
            )

            assert response.status_code == 200
            assert mock_sleep.called  # Should have slept for retry backoff

    def test_circuit_breaker_with_rate_limiting(self):
        """Test circuit breaker combined with rate limiting."""
        rate_limiter = Mock()
        retryable_session = RetryableSession(rate_limiter=rate_limiter)

        # Simulate repeated failures to trigger circuit breaker
        rate_limiter.make_request.side_effect = APIError("Persistent failure")

        # Make calls until circuit opens
        failure_count = 0
        while retryable_session.circuit_breaker.state == CircuitBreaker.CLOSED:
            try:
                retryable_session.request("GET", "https://api.planet.com/test")
            except APIError:
                failure_count += 1
                if failure_count > 10:  # Safety break
                    break

        assert retryable_session.circuit_breaker.state == CircuitBreaker.OPEN

        # Next call should fail fast without calling rate limiter
        rate_limiter.reset_mock()

        with pytest.raises(APIError) as exc_info:
            retryable_session.request("GET", "https://api.planet.com/test")

        assert "Circuit breaker is OPEN" in str(exc_info.value)
        rate_limiter.make_request.assert_not_called()

    def test_complex_retry_scenario(self):
        """Test complex scenario with multiple failure types."""
        rate_limiter = RateLimiter(rates={"test": 10})
        mock_session = Mock()

        # Simulate various failure types followed by success
        responses = [
            ConnectionError("Network error"),  # Retryable
            Mock(status_code=429, headers={"Retry-After": "1"}),  # Rate limited
            Mock(status_code=503),  # Server error
            Mock(status_code=200, headers={}),  # Success
        ]
        mock_session.request.side_effect = responses
        rate_limiter.session = mock_session

        with patch("time.sleep") as mock_sleep:
            response = rate_limiter.make_request(
                "GET", "https://api.planet.com/test/endpoint", max_retries=5
            )

            assert response.status_code == 200
            assert mock_sleep.call_count >= 3  # Should have slept for each retry


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""

    def test_rate_limiter_memory_usage(self):
        """Test rate limiter memory usage with many requests."""
        import sys

        rate_limiter = RateLimiter(rates={"test": 100})
        mock_session = Mock()
        mock_session.request.return_value = Mock(status_code=200, headers={})
        rate_limiter.session = mock_session

        initial_size = sys.getsizeof(rate_limiter)

        # Make many requests
        for i in range(1000):
            rate_limiter.make_request("GET", f"https://api.planet.com/test/{i}")

        final_size = sys.getsizeof(rate_limiter)

        # Memory should not grow excessively
        growth = final_size - initial_size
        assert growth < 100000  # Allow reasonable growth but not excessive

    def test_concurrent_rate_limiting(self):
        """Test that rate limiter handles concurrent requests without errors."""
        import threading
        import queue

        rate_limiter = RateLimiter(rates={"test": 5})
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_session.request.return_value = mock_response
        rate_limiter.session = mock_session

        results = queue.Queue()
        errors = queue.Queue()

        def make_requests():
            """Make multiple requests from this thread."""
            try:
                for i in range(5):
                    start_time = time.time()
                    response = rate_limiter.make_request(
                        "GET", "https://api.planet.com/test"
                    )
                    end_time = time.time()

                    # Verify response is what we expect
                    assert response.status_code == 200

                    duration = end_time - start_time
                    results.put(duration)
            except Exception as e:
                errors.put(e)

        # Start multiple threads to test concurrent access
        threads = []
        num_threads = 3
        requests_per_thread = 5
        total_expected_requests = num_threads * requests_per_thread

        for i in range(num_threads):
            thread = threading.Thread(target=make_requests)
            threads.append(thread)

        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Collect results
        request_times = []
        while not results.empty():
            request_times.append(results.get())

        # Collect any errors
        error_list = []
        while not errors.empty():
            error_list.append(errors.get())

        # SIMPLE ASSERTIONS: Just verify concurrent access works properly

        # No errors should occur during concurrent access
        assert (
            len(error_list) == 0
        ), f"Errors occurred during concurrent access: {error_list}"

        # All requests should complete
        assert (
            len(request_times) == total_expected_requests
        ), f"Expected {total_expected_requests} requests, got {len(request_times)}"

        # All request times should be reasonable (not negative, not too large)
        assert all(
            0 <= t < 10.0 for t in request_times
        ), f"Some request times are unreasonable: {request_times}"

        # Total time should be reasonable (test shouldn't hang)
        assert total_time < 30.0, f"Test took too long: {total_time:.3f}s"

        # Basic functionality verification
        assert all(
            isinstance(t, (int, float)) for t in request_times
        ), "All request times should be numeric"

        print(f"✅ Concurrent access test passed:")
        print(f"   - {len(request_times)} requests completed successfully")
        print(f"   - Total time: {total_time:.3f}s")
        print(
            f"   - Average request time: {sum(request_times) / len(request_times):.6f}s"
        )
        print(f"   - No errors occurred")


class TestErrorScenarios:
    """Test various error scenarios and edge cases."""

    def test_malformed_retry_after_header(self):
        """Test handling of malformed Retry-After headers."""
        rate_limiter = RateLimiter()

        # Test various malformed headers
        malformed_responses = [
            Mock(headers={"Retry-After": "not-a-number"}),
            Mock(headers={"Retry-After": ""}),
            Mock(headers={"Retry-After": "999999999"}),  # Very large number
            Mock(headers={}),  # No header
        ]

        for response in malformed_responses:
            # Should not raise exception
            retry_after = rate_limiter._parse_retry_after(response)
            assert isinstance(retry_after, (int, float))
            assert retry_after >= 0

    def test_extreme_rate_limits(self):
        """Test behavior with extreme rate limit values."""
        # Test very high rate limit
        high_rate_limiter = RateLimiter(rates={"test": 1000})
        assert high_rate_limiter.rates["test"] == 1000

        # Test zero rate limit
        zero_rate_limiter = RateLimiter(rates={"test": 0})
        mock_session = Mock()
        mock_session.request.return_value = Mock(status_code=200, headers={})
        zero_rate_limiter.session = mock_session

        # Should still work (effectively no rate limiting)
        response = zero_rate_limiter.make_request("GET", "https://api.planet.com/test")
        assert response.status_code == 200

    def test_circuit_breaker_edge_cases(self):
        """Test circuit breaker edge cases."""
        # Test with zero failure threshold
        cb = CircuitBreaker(failure_threshold=0)

        def failing_function():
            raise APIError("Failure")

        # Should open immediately on first failure
        with pytest.raises(APIError):
            cb.call(failing_function)

        assert cb.state == CircuitBreaker.OPEN

        # Test with very long recovery timeout
        cb_long_timeout = CircuitBreaker(recovery_timeout=3600)  # 1 hour
        cb_long_timeout.state = CircuitBreaker.OPEN
        cb_long_timeout.last_failure_time = time.time()

        # Should not transition to half-open immediately
        assert not cb_long_timeout._should_attempt_reset()


if __name__ == "__main__":
    pytest.main([__file__])
