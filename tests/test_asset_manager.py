#!/usr/bin/env python3
"""
Tests for planetscope_py.asset_manager module.

This module tests the complete asset management system including:
- Asset status tracking and quota management
- Download job management and progress tracking
- User confirmation workflows and error handling
- Async asset activation and download operations
"""

import pytest
import asyncio
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List

from planetscope_py.asset_manager import (
    AssetManager,
    AssetStatus,
    QuotaInfo,
    DownloadJob,
)
from planetscope_py.auth import PlanetAuth
from planetscope_py.exceptions import AssetError


class TestAssetStatus:
    """Test AssetStatus enum values."""

    def test_asset_status_values(self):
        """Test that all required asset status values exist."""
        expected_statuses = {
            "pending",
            "activating",
            "active",
            "downloading",
            "completed",
            "failed",
            "expired",
        }
        actual_statuses = {status.value for status in AssetStatus}
        assert actual_statuses == expected_statuses


class TestQuotaInfo:
    """Test QuotaInfo dataclass functionality."""

    def test_quota_info_creation(self):
        """Test QuotaInfo object creation with required fields."""
        quota = QuotaInfo(
            current_usage_km2=1000.0,
            monthly_limit_km2=3000.0,
            remaining_km2=2000.0,
            usage_percentage=0.33,
            download_estimate_km2=500.0,
            estimated_scenes_count=10,
        )

        assert quota.current_usage_km2 == 1000.0
        assert quota.monthly_limit_km2 == 3000.0
        assert quota.remaining_km2 == 2000.0
        assert quota.usage_percentage == 0.33
        assert quota.download_estimate_km2 == 500.0
        assert quota.estimated_scenes_count == 10
        assert quota.warning_threshold == 0.8  # Default value

    def test_is_near_limit(self):
        """Test is_near_limit property calculation."""
        # Below threshold
        quota_low = QuotaInfo(
            current_usage_km2=1000.0,
            monthly_limit_km2=3000.0,
            remaining_km2=2000.0,
            usage_percentage=0.33,
            download_estimate_km2=0.0,
            estimated_scenes_count=0,
        )
        assert not quota_low.is_near_limit

        # Above threshold
        quota_high = QuotaInfo(
            current_usage_km2=2500.0,
            monthly_limit_km2=3000.0,
            remaining_km2=500.0,
            usage_percentage=0.83,
            download_estimate_km2=0.0,
            estimated_scenes_count=0,
        )
        assert quota_high.is_near_limit

    def test_can_download(self):
        """Test can_download property calculation."""
        # Can download
        quota_can = QuotaInfo(
            current_usage_km2=1000.0,
            monthly_limit_km2=3000.0,
            remaining_km2=2000.0,
            usage_percentage=0.33,
            download_estimate_km2=500.0,
            estimated_scenes_count=5,
        )
        assert quota_can.can_download

        # Cannot download (would exceed limit)
        quota_cannot = QuotaInfo(
            current_usage_km2=2000.0,
            monthly_limit_km2=3000.0,
            remaining_km2=1000.0,
            usage_percentage=0.67,
            download_estimate_km2=1500.0,
            estimated_scenes_count=15,
        )
        assert not quota_cannot.can_download


class TestDownloadJob:
    """Test DownloadJob dataclass functionality."""

    def test_download_job_creation(self):
        """Test DownloadJob object creation with required fields."""
        job = DownloadJob(
            scene_id="test_scene_001",
            asset_type="ortho_analytic_4b",
            item_type="PSScene",
        )

        assert job.scene_id == "test_scene_001"
        assert job.asset_type == "ortho_analytic_4b"
        assert job.item_type == "PSScene"
        assert job.status == AssetStatus.PENDING
        assert job.retry_count == 0
        assert job.max_retries == 3

    def test_duration_calculation(self):
        """Test duration_seconds property calculation."""
        job = DownloadJob("scene_001", "ortho_analytic_4b")

        # No times set
        assert job.duration_seconds is None

        # Set activation and completion times
        job.activation_time = datetime(2025, 1, 1, 12, 0, 0)
        job.completion_time = datetime(2025, 1, 1, 12, 5, 30)

        expected_duration = 330.0  # 5 minutes 30 seconds
        assert job.duration_seconds == expected_duration

    def test_is_expired(self):
        """Test is_expired property calculation."""
        job = DownloadJob("scene_001", "ortho_analytic_4b")

        # Not active, should not be expired
        assert not job.is_expired

        # Active but recent
        job.status = AssetStatus.ACTIVE
        job.activation_time = datetime.now() - timedelta(hours=1)
        assert not job.is_expired

        # Active but old (expired)
        job.activation_time = datetime.now() - timedelta(hours=5)
        assert job.is_expired


class TestAssetManager:
    """Test AssetManager main functionality."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock PlanetAuth for testing."""
        auth = Mock(spec=PlanetAuth)
        session = Mock()
        auth.get_session.return_value = session
        return auth

    @pytest.fixture
    def asset_manager(self, mock_auth):
        """Create AssetManager instance for testing."""
        with patch("planetscope_py.asset_manager.RateLimiter"):
            return AssetManager(mock_auth)

    def test_asset_manager_initialization(self, mock_auth):
        """Test AssetManager initialization with proper configuration."""
        with patch("planetscope_py.asset_manager.RateLimiter") as mock_rate_limiter:
            config = {
                "max_concurrent_downloads": 10,
                "asset_types": ["ortho_analytic_4b", "ortho_visual"],
                "download_chunk_size": 16384,
            }

            manager = AssetManager(mock_auth, config)

            assert manager.auth == mock_auth
            assert manager.max_concurrent_downloads == 10
            assert manager.default_asset_types == ["ortho_analytic_4b", "ortho_visual"]
            assert manager.chunk_size == 16384

            # Verify RateLimiter was initialized
            mock_rate_limiter.assert_called_once()

    # FIXED: Remove @patch decorator that was causing the test failure
    def test_estimate_download_impact(self, asset_manager):
        """Test download impact estimation."""
        # Mock scenes
        scenes = [
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                }
            }
        ]

        # FIXED: Call the method directly since it's now synchronous
        result = asset_manager.estimate_download_impact(scenes)

        # FIXED: Now properly check that result is QuotaInfo instance
        assert isinstance(result, QuotaInfo)
        assert result.estimated_scenes_count == 1
        assert result.download_estimate_km2 > 0

    @patch("builtins.input", return_value="n")
    def test_get_user_confirmation_exceeds_quota(self, mock_input, asset_manager):
        """Test user confirmation when download exceeds quota."""
        quota_info = QuotaInfo(
            current_usage_km2=2800.0,
            monthly_limit_km2=3000.0,
            remaining_km2=200.0,
            usage_percentage=0.93,
            download_estimate_km2=500.0,  # Would exceed limit
            estimated_scenes_count=5,
        )

        result = asset_manager.get_user_confirmation(quota_info)
        assert result is False  # Should automatically refuse

    @patch("builtins.input", return_value="y")
    def test_get_user_confirmation_user_accepts(self, mock_input, asset_manager):
        """Test user confirmation when user accepts download."""
        quota_info = QuotaInfo(
            current_usage_km2=1000.0,
            monthly_limit_km2=3000.0,
            remaining_km2=2000.0,
            usage_percentage=0.33,
            download_estimate_km2=500.0,
            estimated_scenes_count=5,
        )

        result = asset_manager.get_user_confirmation(quota_info)
        assert result is True
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="n")
    def test_get_user_confirmation_user_declines(self, mock_input, asset_manager):
        """Test user confirmation when user declines download."""
        quota_info = QuotaInfo(
            current_usage_km2=1000.0,
            monthly_limit_km2=3000.0,
            remaining_km2=2000.0,
            usage_percentage=0.33,
            download_estimate_km2=500.0,
            estimated_scenes_count=5,
        )

        result = asset_manager.get_user_confirmation(quota_info)
        assert result is False
        mock_input.assert_called_once()

    def test_get_download_status_no_jobs(self, asset_manager):
        """Test download status when no jobs are active."""
        status = asset_manager.get_download_status()

        assert status["status"] == "no_jobs"
        assert "No download jobs active" in status["message"]

    def test_get_download_status_with_jobs(self, asset_manager):
        """Test download status with active jobs."""
        # Add some mock jobs
        asset_manager.download_jobs = [
            DownloadJob("scene_001", "ortho_analytic_4b"),
            DownloadJob("scene_002", "ortho_analytic_4b"),
        ]
        asset_manager.download_jobs[0].status = AssetStatus.COMPLETED
        asset_manager.download_jobs[0].file_size_mb = 25.5
        asset_manager.download_jobs[1].status = AssetStatus.FAILED
        asset_manager.download_jobs[1].error_message = "Download timeout"

        status = asset_manager.get_download_status()

        assert status["total_jobs"] == 2
        assert status["status_breakdown"]["completed"] == 1
        assert status["status_breakdown"]["failed"] == 1
        assert status["completed_size_mb"] == 25.5
        assert len(status["failed_jobs"]) == 1
        assert status["failed_jobs"][0]["error"] == "Download timeout"

    def test_export_download_report(self, asset_manager):
        """Test download report export functionality."""
        # Add mock job
        asset_manager.download_jobs = [DownloadJob("scene_001", "ortho_analytic_4b")]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            report_path = f.name

        try:
            asset_manager.export_download_report(report_path)

            # Verify report was created
            assert Path(report_path).exists()

            # Verify report content
            with open(report_path, "r") as f:
                report = json.load(f)

            assert "download_summary" in report
            assert "job_details" in report
            assert "generated_at" in report
            assert len(report["job_details"]) == 1

        finally:
            Path(report_path).unlink(missing_ok=True)


class TestAssetManagerAsync:
    """Test AssetManager async functionality."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock PlanetAuth for testing."""
        auth = Mock(spec=PlanetAuth)
        session = Mock()
        auth.get_session.return_value = session
        return auth

    @pytest.fixture
    def asset_manager(self, mock_auth):
        """Create AssetManager instance for testing."""
        with patch("planetscope_py.asset_manager.RateLimiter"):
            return AssetManager(mock_auth)

    @pytest.mark.asyncio
    async def test_check_user_quota_success(self, asset_manager):
        """Test successful quota checking."""
        with patch.object(
            asset_manager, "_get_quota_from_multiple_sources"
        ) as mock_quota:
            mock_quota.return_value = {
                "current_usage_km2": 1500.0,
                "monthly_limit_km2": 3000.0,
            }

            quota_info = await asset_manager.check_user_quota()

            assert isinstance(quota_info, QuotaInfo)
            assert quota_info.current_usage_km2 == 1500.0
            assert quota_info.monthly_limit_km2 == 3000.0
            assert quota_info.remaining_km2 == 1500.0
            assert quota_info.usage_percentage == 0.5

    @pytest.mark.asyncio
    async def test_check_user_quota_failure(self, asset_manager):
        """Test quota checking with API failure (fallback to estimated)."""
        with patch.object(
            asset_manager, "_get_quota_from_multiple_sources"
        ) as mock_quota:
            mock_quota.side_effect = Exception("API Error")

            quota_info = await asset_manager.check_user_quota()

            # Should return estimated quota
            assert isinstance(quota_info, QuotaInfo)
            assert quota_info.current_usage_km2 == 0.0
            assert quota_info.monthly_limit_km2 == 3000.0

    @pytest.mark.asyncio
    async def test_get_quota_from_analytics_api(self, asset_manager):
        """Test quota retrieval from Analytics API."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "subscriptions": [
                {
                    "plan_id": "planetscope-basic",
                    "quota": {"area_limit_km2": 3000.0, "area_used_km2": 1200.0},
                }
            ]
        }

        asset_manager.rate_limiter.make_request = Mock(return_value=mock_response)

        result = await asset_manager._get_quota_from_analytics_api()

        assert result is not None
        assert result["monthly_limit_km2"] == 3000.0
        assert result["current_usage_km2"] == 1200.0
        assert result["source"] == "analytics_api"

    @pytest.mark.asyncio
    async def test_activate_and_download_assets_cancelled_by_user(self, asset_manager):
        """Test workflow when user cancels download."""
        scenes = [
            {
                "properties": {"id": "scene_001"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ]

        with patch.object(asset_manager, "get_user_confirmation", return_value=False):
            result = await asset_manager.activate_and_download_assets(scenes)

            assert result == []  # Empty list when cancelled
            assert len(asset_manager.download_jobs) == 0

    @pytest.mark.asyncio
    async def test_activate_assets_batch(self, asset_manager):
        """Test batch asset activation."""
        jobs = [
            DownloadJob("scene_001", "ortho_analytic_4b"),
            DownloadJob("scene_002", "ortho_analytic_4b"),
        ]

        # Mock successful activation responses
        mock_response_assets = Mock()
        mock_response_assets.status_code = 200
        mock_response_assets.json.return_value = {
            "ortho_analytic_4b": {
                "status": "inactive",
                "_links": {"activate": "https://api.planet.com/activate/123"},
            }
        }

        mock_response_activate = Mock()
        mock_response_activate.status_code = 202

        asset_manager.rate_limiter.make_request = Mock(
            side_effect=[mock_response_assets, mock_response_activate] * 2
        )

        await asset_manager._activate_assets_batch(jobs)

        # Should have made activation requests
        assert (
            asset_manager.rate_limiter.make_request.call_count == 4
        )  # 2 assets × 2 calls each

    @pytest.mark.asyncio
    async def test_download_single_asset_success(self, asset_manager):
        """Test successful single asset download."""
        job = DownloadJob("scene_001", "ortho_analytic_4b")
        job.download_url = "http://example.com/download"
        job.status = AssetStatus.ACTIVE

        output_path = Path(tempfile.mkdtemp())

        # Create a mock that properly handles the aiohttp session context
        class MockResponse:
            def __init__(self):
                self.status = 200
                self.headers = {"content-length": "1024"}
                self.content = Mock()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        class MockContent:
            async def iter_chunked(self, size):
                yield b"test_data" * 128  # 1024 bytes total

        mock_response = MockResponse()
        mock_response.content = MockContent()

        class MockSession:
            def get(self, url):
                return mock_response

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        with patch("aiohttp.ClientSession", MockSession):
            await asset_manager._download_single_asset(job, output_path)

            assert job.status == AssetStatus.COMPLETED
            assert job.file_path is not None
            assert job.file_size_mb is not None
            assert job.completion_time is not None

    @pytest.mark.asyncio
    async def test_download_single_asset_failure_with_retry(self, asset_manager):
        """Test download failure with retry logic."""
        job = DownloadJob("scene_001", "ortho_analytic_4b")
        job.download_url = "http://example.com/download"
        job.status = AssetStatus.ACTIVE
        job.max_retries = 1  # Limit retries for test

        output_path = Path(tempfile.mkdtemp())

        # Create a mock that properly handles the aiohttp session context for failure
        class MockResponse:
            def __init__(self):
                self.status = 500
                self.headers = {"content-length": "1024"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        mock_response = MockResponse()

        class MockSession:
            def get(self, url):
                return mock_response

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        with patch("aiohttp.ClientSession", MockSession):
            with patch("asyncio.sleep"):  # Speed up test
                await asset_manager._download_single_asset(job, output_path)

            assert job.status == AssetStatus.FAILED
            assert job.retry_count == job.max_retries
            assert "Download failed: HTTP 500" in job.error_message


class TestAssetManagerIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock PlanetAuth for testing."""
        auth = Mock(spec=PlanetAuth)
        session = Mock()
        auth.get_session.return_value = session
        return auth

    @pytest.mark.asyncio
    async def test_full_workflow_mock(self, mock_auth):
        """Test complete workflow with mocked API responses."""
        with patch("planetscope_py.asset_manager.RateLimiter"):
            asset_manager = AssetManager(mock_auth)

        # Mock scenes
        scenes = [
            {
                "properties": {"id": "scene_001"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ]

        # Mock user confirmation
        with patch.object(asset_manager, "get_user_confirmation", return_value=True):
            with patch.object(asset_manager, "_activate_assets_batch") as mock_activate:
                with patch.object(
                    asset_manager, "_download_activated_assets"
                ) as mock_download:
                    with patch.object(asset_manager, "_print_download_summary"):

                        result = await asset_manager.activate_and_download_assets(
                            scenes, confirm_download=True, output_dir=tempfile.mkdtemp()
                        )

                        # Verify workflow steps were called
                        mock_activate.assert_called_once()
                        mock_download.assert_called_once()

                        # Verify jobs were created
                        assert len(result) == 1
                        assert result[0].scene_id == "scene_001"
                        assert result[0].asset_type == "ortho_analytic_4b"


if __name__ == "__main__":
    pytest.main([__file__])
