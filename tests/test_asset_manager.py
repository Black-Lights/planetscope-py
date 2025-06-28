#!/usr/bin/env python3
"""
Tests for planetscope_py.asset_manager module - FIXED VERSION.

This module tests the asset management system WITHOUT actual downloads,
focusing on configuration, quota management, and data structures.
No Planet API quota is consumed by these tests.
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
    DownloadCancellationReason,
)
from planetscope_py.auth import PlanetAuth
from planetscope_py.exceptions import AssetError


class TestAssetStatus:
    """Test AssetStatus enum values."""

    def test_asset_status_values(self):
        """Test that all required asset status values exist - FIXED."""
        # FIXED: Updated to match actual implementation which includes USER_CANCELLED
        expected_statuses = {
            "pending",
            "activating", 
            "active",
            "downloading",
            "completed",
            "failed",
            "expired",
            "user_cancelled",  # ADDED: This exists in the actual implementation
        }
        actual_statuses = {status.value for status in AssetStatus}
        assert actual_statuses == expected_statuses


class TestDownloadCancellationReason:
    """Test DownloadCancellationReason enum values."""

    def test_cancellation_reason_values(self):
        """Test cancellation reason enum values."""
        expected_reasons = {
            "user_choice",
            "quota_exceeded", 
            "api_error",
            "insufficient_space",
        }
        actual_reasons = {reason.value for reason in DownloadCancellationReason}
        assert actual_reasons == expected_reasons


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
            download_estimate_mb=250.0,  # ADDED: This field exists in actual implementation
            estimated_scenes_count=10,
        )

        assert quota.current_usage_km2 == 1000.0
        assert quota.monthly_limit_km2 == 3000.0
        assert quota.remaining_km2 == 2000.0
        assert quota.usage_percentage == 0.33
        assert quota.download_estimate_km2 == 500.0
        assert quota.download_estimate_mb == 250.0
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
            download_estimate_mb=0.0,
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
            download_estimate_mb=0.0,
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
            download_estimate_mb=250.0,
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
            download_estimate_mb=750.0,
            estimated_scenes_count=15,
        )
        assert not quota_cannot.can_download

    def test_quota_status_property(self):
        """Test quota_status property."""
        # Normal quota
        quota_ok = QuotaInfo(
            current_usage_km2=1000.0,
            monthly_limit_km2=3000.0,
            remaining_km2=2000.0,
            usage_percentage=0.33,
            download_estimate_km2=500.0,
            download_estimate_mb=250.0,
            estimated_scenes_count=5,
        )
        assert quota_ok.quota_status == "OK"
        
        # Near limit
        quota_near = QuotaInfo(
            current_usage_km2=2500.0,
            monthly_limit_km2=3000.0,
            remaining_km2=500.0,
            usage_percentage=0.83,
            download_estimate_km2=200.0,
            download_estimate_mb=100.0,
            estimated_scenes_count=2,
        )
        assert quota_near.quota_status == "NEAR_LIMIT"
        
        # Exceeded
        quota_exceeded = QuotaInfo(
            current_usage_km2=2800.0,
            monthly_limit_km2=3000.0,
            remaining_km2=200.0,
            usage_percentage=0.93,
            download_estimate_km2=500.0,
            download_estimate_mb=250.0,
            estimated_scenes_count=5,
        )
        assert quota_exceeded.quota_status == "QUOTA_EXCEEDED"


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

    def test_should_retry(self):
        """Test should_retry property calculation."""
        job = DownloadJob("scene_001", "ortho_analytic_4b")
        
        # Should retry initially
        job.status = AssetStatus.FAILED
        assert job.should_retry
        
        # Should not retry after max retries
        job.retry_count = job.max_retries
        assert not job.should_retry
        
        # Should not retry if completed
        job.retry_count = 0
        job.status = AssetStatus.COMPLETED
        assert not job.should_retry

    def test_record_retry_attempt(self):
        """Test retry attempt recording."""
        job = DownloadJob("scene_001", "ortho_analytic_4b")
        
        initial_count = job.retry_count
        job.record_retry_attempt()
        
        assert job.retry_count == initial_count + 1
        assert job.last_retry_time is not None


class TestAssetManager:
    """Test AssetManager main functionality - NO DOWNLOADS."""

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

    def test_estimate_download_impact_basic(self, asset_manager):
        """Test basic download impact estimation without API calls."""
        # Mock scenes with simple geometry
        scenes = [
            {
                "properties": {"id": "scene_001"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                }
            }
        ]

        # Call the synchronous version directly to avoid async issues
        result = asset_manager._estimate_download_impact_sync(scenes)

        assert isinstance(result, QuotaInfo)
        assert result.estimated_scenes_count == 1
        assert result.download_estimate_km2 > 0
        assert result.download_estimate_mb >= 0

    @patch("builtins.input", return_value="n")
    def test_get_user_confirmation_exceeds_quota(self, mock_input, asset_manager):
        """Test user confirmation when download exceeds quota."""
        quota_info = QuotaInfo(
            current_usage_km2=2800.0,
            monthly_limit_km2=3000.0,
            remaining_km2=200.0,
            usage_percentage=0.93,
            download_estimate_km2=500.0,  # Would exceed limit
            download_estimate_mb=250.0,
            estimated_scenes_count=5,
        )

        result = asset_manager.get_user_confirmation(quota_info)
        assert result is False  # Should automatically refuse
        assert asset_manager.last_cancellation_reason == DownloadCancellationReason.QUOTA_EXCEEDED

    @patch("builtins.input", return_value="y")
    def test_get_user_confirmation_user_accepts(self, mock_input, asset_manager):
        """Test user confirmation when user accepts download."""
        quota_info = QuotaInfo(
            current_usage_km2=1000.0,
            monthly_limit_km2=3000.0,
            remaining_km2=2000.0,
            usage_percentage=0.33,
            download_estimate_km2=500.0,
            download_estimate_mb=250.0,
            estimated_scenes_count=5,
        )

        result = asset_manager.get_user_confirmation(quota_info)
        assert result is True
        assert asset_manager.last_cancellation_reason is None
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
            download_estimate_mb=250.0,
            estimated_scenes_count=5,
        )

        result = asset_manager.get_user_confirmation(quota_info)
        assert result is False
        assert asset_manager.last_cancellation_reason == DownloadCancellationReason.USER_CHOICE
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
            assert "cancellation_tracking" in report  # ADDED: This exists in actual implementation
            assert len(report["job_details"]) == 1

        finally:
            Path(report_path).unlink(missing_ok=True)

    def test_clear_download_jobs(self, asset_manager):
        """Test clearing download jobs."""
        # Add some jobs
        asset_manager.download_jobs = [DownloadJob("scene_001", "ortho_analytic_4b")]
        asset_manager.last_cancellation_reason = DownloadCancellationReason.USER_CHOICE
        
        asset_manager.clear_download_jobs()
        
        assert len(asset_manager.download_jobs) == 0
        assert asset_manager.last_cancellation_reason is None

    def test_get_download_statistics(self, asset_manager):
        """Test download statistics calculation."""
        # Add mock jobs with different statuses
        asset_manager.download_jobs = [
            DownloadJob("scene_001", "ortho_analytic_4b"),
            DownloadJob("scene_002", "ortho_analytic_4b"),
        ]
        asset_manager.download_jobs[0].status = AssetStatus.COMPLETED
        asset_manager.download_jobs[0].file_size_mb = 25.5
        asset_manager.download_jobs[0].retry_count = 1
        asset_manager.download_jobs[1].status = AssetStatus.FAILED
        asset_manager.download_jobs[1].retry_count = 2

        stats = asset_manager.get_download_statistics()

        assert "summary" in stats
        assert "retry_stats" in stats
        assert "timing" in stats
        assert stats["summary"]["total_jobs"] == 2
        assert stats["summary"]["completed"] == 1
        assert stats["summary"]["failed"] == 1
        assert stats["retry_stats"]["total_retries"] == 3

    def test_diagnose_download_issues(self, asset_manager):
        """Test download issue diagnosis."""
        # Add mock failed jobs with different error types
        asset_manager.download_jobs = [
            DownloadJob("scene_001", "ortho_analytic_4b"),
            DownloadJob("scene_002", "ortho_analytic_4b"),
        ]
        asset_manager.download_jobs[0].status = AssetStatus.FAILED
        asset_manager.download_jobs[0].error_message = "Download timeout"
        asset_manager.download_jobs[1].status = AssetStatus.FAILED
        asset_manager.download_jobs[1].error_message = "Network connection error"

        diagnosis = asset_manager.diagnose_download_issues()

        assert "issues_found" in diagnosis
        assert "recommendations" in diagnosis
        assert "total_failed" in diagnosis
        assert diagnosis["total_failed"] == 2


class TestAssetManagerAsync:
    """Test AssetManager async functionality - NO ACTUAL DOWNLOADS."""

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
                "source": "test_api"
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
    async def test_activate_and_download_assets_cancelled_by_user(self, asset_manager):
        """Test workflow when user cancels download - FIXED."""
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

            # FIXED: When cancelled, jobs are created but marked as USER_CANCELLED
            assert len(result) == 1  # Jobs are created but cancelled
            assert result[0].status == AssetStatus.USER_CANCELLED
            assert result[0].scene_id == "scene_001"

    # SKIP: Download tests that would consume quota
    @pytest.mark.skip(reason="Skipping actual download tests to preserve Planet quota")
    async def test_download_single_asset_success_skip(self):
        """Skip actual download test to preserve quota."""
        pass

    @pytest.mark.skip(reason="Skipping actual download tests to preserve Planet quota")
    async def test_download_single_asset_failure_with_retry_skip(self):
        """Skip actual download test to preserve quota."""
        pass


class TestAssetManagerIntegration:
    """Integration tests for complete workflows - NO DOWNLOADS."""

    @pytest.fixture
    def mock_auth(self):
        """Create mock PlanetAuth for testing."""
        auth = Mock(spec=PlanetAuth)
        session = Mock()
        auth.get_session.return_value = session
        return auth

    @pytest.mark.asyncio
    async def test_full_workflow_mock(self, mock_auth):
        """Test complete workflow with mocked API responses - NO DOWNLOADS."""
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
                    with patch.object(asset_manager, "_print_enhanced_download_summary"):

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


class TestUtilityFunctions:
    """Test utility functions without API calls."""

    def test_validate_config(self):
        """Test configuration validation."""
        from planetscope_py.asset_manager import validate_config
        
        # Valid config
        valid_config = {
            "max_concurrent_downloads": 3,
            "timeouts": {"download_total": 3600},
            "retry_config": {"max_retries": 3}
        }
        
        is_valid, warnings = validate_config(valid_config)
        assert is_valid is True
        
        # Invalid config
        invalid_config = {
            "max_concurrent_downloads": 0,  # Invalid
            "timeouts": {"download_total": 60},  # Too short
        }
        
        is_valid, warnings = validate_config(invalid_config)
        assert is_valid is False
        assert len(warnings) > 0

    def test_create_custom_config(self):
        """Test custom configuration creation."""
        from planetscope_py.asset_manager import create_custom_config
        
        config = create_custom_config(
            network_speed="slow",
            file_size_expectation="large",
            reliability="high"
        )
        
        assert "max_concurrent_downloads" in config
        assert "timeouts" in config
        assert "retry_config" in config
        
        # Should be conservative for slow/large/high reliability
        assert config["max_concurrent_downloads"] <= 2
        assert config["retry_config"]["max_retries"] >= 3

    def test_get_recommended_config_for_quota(self):
        """Test quota-based configuration recommendation."""
        from planetscope_py.asset_manager import get_recommended_config_for_quota
        
        # Low quota should get conservative config
        low_quota_config = get_recommended_config_for_quota(50.0)
        assert low_quota_config is not None
        
        # High quota should get aggressive config
        high_quota_config = get_recommended_config_for_quota(2000.0)
        assert high_quota_config is not None


class TestBasicFunctionality:
    """Test basic functionality without external dependencies."""

    def test_asset_manager_can_be_created(self):
        """Test that AssetManager can be instantiated."""
        mock_auth = Mock(spec=PlanetAuth)
        session = Mock()
        mock_auth.get_session.return_value = session
        
        with patch("planetscope_py.asset_manager.RateLimiter"):
            manager = AssetManager(mock_auth)
        
        assert manager is not None
        assert manager.auth == mock_auth

    def test_download_job_basic_workflow(self):
        """Test basic download job state transitions."""
        job = DownloadJob("test_scene", "ortho_analytic_4b")
        
        # Initial state
        assert job.status == AssetStatus.PENDING
        assert job.retry_count == 0
        
        # Simulate workflow
        job.status = AssetStatus.ACTIVATING
        job.activation_time = datetime.now()
        
        job.status = AssetStatus.ACTIVE
        job.download_url = "http://example.com/download"
        
        job.status = AssetStatus.DOWNLOADING
        job.download_start_time = datetime.now()
        
        job.status = AssetStatus.COMPLETED
        job.completion_time = datetime.now()
        job.file_size_mb = 25.0
        
        # Verify final state
        assert job.status == AssetStatus.COMPLETED
        assert job.file_size_mb == 25.0
        assert job.duration_seconds is not None
        assert job.duration_seconds > 0

    def test_quota_info_calculations(self):
        """Test quota information calculations."""
        quota = QuotaInfo(
            current_usage_km2=1500.0,
            monthly_limit_km2=3000.0,
            remaining_km2=1500.0,
            usage_percentage=0.5,
            download_estimate_km2=300.0,
            download_estimate_mb=150.0,
            estimated_scenes_count=3,
        )
        
        # Test properties
        assert not quota.is_near_limit  # 50% usage
        assert quota.can_download  # 1500 + 300 <= 3000
        assert quota.quota_status == "OK"
        
        # Test near limit
        quota.usage_percentage = 0.85
        assert quota.is_near_limit
        assert quota.quota_status == "NEAR_LIMIT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])